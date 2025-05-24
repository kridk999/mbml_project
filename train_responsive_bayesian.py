#!/usr/bin/env python3
"""
Train Responsive Bayesian LSTM
==============================

Trains the responsive Bayesian LSTM model, which combines event sensitivity
with uncertainty quantification.
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import the responsive Bayesian model
from responsive_bayesian_lstm import ResponsiveBayesianLSTM

# Define constants
DEFAULT_FEATURE_COLS = [
    "tick", "numCTAlive", "numTAlive", "ctEquipValue", "tEquipValue", 
    "bombPlanted", "ctWon", "tWon"
]

def debug_parameter_distributions():
    """Debug function to inspect parameter distributions."""
    print("\n" + "="*50)
    print("PARAMETER STORE DEBUGGING")
    print("="*50)
    
    param_store = pyro.get_param_store()
    
    for name, param in param_store.items():
        if isinstance(param, torch.Tensor):
            print(f"\nParameter: {name}")
            print(f"  Shape: {param.shape}")
            print(f"  Mean: {param.mean().item():.4f}")
            print(f"  Std: {param.std().item():.4f}")
            print(f"  Min: {param.min().item():.4f}")
            print(f"  Max: {param.max().item():.4f}")
            
            # Check for gradients
            if param.grad is not None:
                print(f"  Grad norm: {param.grad.norm().item():.4f}")
    
    print("\n" + "="*50)
    
    # Return the parameter store for interactive debugging
    return param_store

def process_features(dataframe, feature_cols=None):
    """Extract features and identify special feature indices."""
    
    if feature_cols is None:
        # Use all numeric columns except specific metadata
        exclude_cols = ["match_id", "round_idx", "tick", "rnd_winner", "team1_score", "team2_score"]
        numeric_cols = dataframe.select_dtypes(include=["number"]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]

    # Identify player count features 
    player_count_cols = [col for col in feature_cols if "alive" in col.lower() or "players" in col.lower()]
    player_count_indices = [feature_cols.index(col) for col in player_count_cols]
    
    # Identify equipment value features
    equipment_cols = [col for col in feature_cols if "eqval" in col.lower() or "equipment" in col.lower()]
    equipment_indices = [feature_cols.index(col) for col in equipment_cols]
    
    print(f"Found {len(feature_cols)} total features")
    print(f"Found {len(player_count_indices)} player count features: {player_count_cols}")
    print(f"Found {len(equipment_indices)} equipment features: {equipment_cols}")

    return feature_cols, player_count_indices, equipment_indices


def prepare_data(csv_path, train_ratio=0.8, feature_cols=None):
    """Load and prepare data for training."""
    print(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows")
    
    # Process features
    feature_cols, player_count_indices, equipment_indices = process_features(df, feature_cols)
    
    # Extract features and labels
    features = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    
    # Convert labels (0 = CT win, 1 = T win)
    labels = []
    for winner in df["rnd_winner"]:
        if winner == "CT":
            labels.append(0.0)
        else:
            labels.append(1.0)
    
    labels = torch.tensor(labels, dtype=torch.float32)
    
    # Split data into rounds
    unique_matches = df["match_id"].unique()
    unique_rounds = []
    
    for match_id in unique_matches:
        match_rounds = df[df["match_id"] == match_id]["round_idx"].unique()
        for round_idx in match_rounds:
            unique_rounds.append((match_id, round_idx))
    
    print(f"Found {len(unique_rounds)} unique rounds")
    
    # Split into train and test sets
    np.random.shuffle(unique_rounds)
    train_size = int(len(unique_rounds) * train_ratio)
    train_rounds = unique_rounds[:train_size]
    test_rounds = unique_rounds[train_size:]
    
    print(f"Train set: {len(train_rounds)} rounds")
    print(f"Test set: {len(test_rounds)} rounds")
    
    # Create train and test datasets
    train_features = []
    train_labels = []
    test_features = []
    test_labels = []
    
    for match_id, round_idx in train_rounds:
        round_df = df[(df["match_id"] == match_id) & (df["round_idx"] == round_idx)]
        round_features = torch.tensor(round_df[feature_cols].values, dtype=torch.float32)
        
        # Get the winner for this round
        winner = round_df.iloc[0]["rnd_winner"]
        round_label = 0.0 if winner == "CT" else 1.0
        round_labels = torch.tensor([round_label] * len(round_df), dtype=torch.float32)
        
        train_features.append(round_features)
        train_labels.append(round_labels)
    
    for match_id, round_idx in test_rounds:
        round_df = df[(df["match_id"] == match_id) & (df["round_idx"] == round_idx)]
        round_features = torch.tensor(round_df[feature_cols].values, dtype=torch.float32)
        
        # Get the winner for this round
        winner = round_df.iloc[0]["rnd_winner"]
        round_label = 0.0 if winner == "CT" else 1.0
        round_labels = torch.tensor([round_label] * len(round_df), dtype=torch.float32)
        
        test_features.append(round_features)
        test_labels.append(round_labels)
    
    return (train_features, train_labels), (test_features, test_labels), (player_count_indices, equipment_indices)


def train_model(train_data, test_data, feature_indices, args):
    """Train the responsive Bayesian LSTM model."""
    (train_features, train_labels), (test_features, test_labels), (player_count_indices, equipment_indices) = train_data, test_data, feature_indices
    
    # Extract input dimension from the first feature set
    input_dim = train_features[0].shape[1] if train_features else 0
    
    if input_dim == 0:
        print("Error: No training data available.")
        return None
    
    print(f"Input dimension: {input_dim}")
    print(f"Player count indices: {player_count_indices}")
    print(f"Equipment indices: {equipment_indices}")
    
    # Create the model
    model = ResponsiveBayesianLSTM(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        player_count_indices=player_count_indices,
        equipment_indices=equipment_indices,
        use_entmax=args.use_entmax
    )
    
    # Set up Pyro optimizer and SVI
    pyro.clear_param_store()
    adam = Adam({"lr": args.learning_rate})
    elbo = Trace_ELBO()
    svi = SVI(model.model, model.guide, adam, loss=elbo)
    
    # Create directory for saving model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        # Train
        epoch_train_loss = 0.0
        num_train_batches = 0
        
        for features, labels in tqdm(zip(train_features, train_labels), desc=f"Epoch {epoch+1}/{args.num_epochs}"):
            # Add batch dimension
            features = features.unsqueeze(0)
            labels = labels.unsqueeze(0)
            
            # Compute loss
            loss = svi.step(features, labels)
            epoch_train_loss += loss
            num_train_batches += 1
        
        avg_train_loss = epoch_train_loss / num_train_batches if num_train_batches > 0 else float('inf')
        train_losses.append(avg_train_loss)
        
        # Test
        epoch_test_loss = 0.0
        num_test_batches = 0
        
        for features, labels in tqdm(zip(test_features, test_labels), desc="Testing"):
            # Add batch dimension
            features = features.unsqueeze(0)
            labels = labels.unsqueeze(0)
            
            # Compute loss
            loss = svi.evaluate_loss(features, labels)
            epoch_test_loss += loss
            num_test_batches += 1
        
        avg_test_loss = epoch_test_loss / num_test_batches if num_test_batches > 0 else float('inf')
        test_losses.append(avg_test_loss)
        
        print(f"Epoch {epoch+1}/{args.num_epochs}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")
        
        # Save the best model
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            
            # Save model checkpoint
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'test_loss': avg_test_loss,
                'input_dim': input_dim,
                'hidden_dim': args.hidden_dim,
                'num_layers': args.num_layers,
                'dropout': args.dropout,
                'player_count_indices': player_count_indices,
                'equipment_indices': equipment_indices,
                'use_entmax': args.use_entmax
            }
            
            torch.save(checkpoint, output_dir / "best_model.pt")
            print(f"Saved best model (test loss: {avg_test_loss:.4f})")
    
    # Plot training and test loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Test Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "training_loss.png")
    plt.close()
    
    # Save final model
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': args.num_epochs,
        'train_loss': train_losses[-1],
        'test_loss': test_losses[-1],
        'input_dim': input_dim,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'player_count_indices': player_count_indices,
        'equipment_indices': equipment_indices,
        'use_entmax': args.use_entmax
    }
    
    torch.save(checkpoint, output_dir / "final_model.pt")
    print(f"Saved final model")
    
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Responsive Bayesian LSTM model")
    parser.add_argument("--csv_path", type=str, default="round_frame_data1.csv", help="Path to CSV data file")
    parser.add_argument("--output_dir", type=str, default="./saved_models/responsive_bayesian", help="Output directory for model")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension size")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of data to use for training")
    parser.add_argument("--use_entmax", action="store_true", help="Use entmax activation (vs sigmoid)")
    
    args = parser.parse_args()
    
    # Prepare data
    train_data, test_data, feature_indices = prepare_data(args.csv_path, args.train_ratio)
    
    # Train model
    model = train_model(train_data, test_data, feature_indices, args)
    
    if model:
        print("Training completed successfully!")
    else:
        print("Training failed.")
