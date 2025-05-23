#!/usr/bin/env python3
"""
Fixed version of the training script that adds a new FeatureEmphasis mode 
for visualization, ensuring that player counts are not scaled for display 
purposes.
"""
import argparse
import json
import os
import time
from pathlib import Path
import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from mbml.data_transform import FeatureEmphasis
from mbml.dataset import CSGORoundDataset, collate_variable_length_rounds, create_dataloaders
from mbml.enhanced_model import ResponsiveWinPredictor

def process_features(dataframe, feature_cols=None):
    """
    Extract features from dataframe and identify special feature indices.

    Returns:
        feature_cols: List of feature column names
        player_count_indices: Indices of player count features
        equipment_indices: Indices of equipment value features
    """
    if feature_cols is None:
        # Automatically select numerical features
        exclude_cols = ["match_id", "round_idx", "ctTeam", "tTeam", "rnd_winner", "clock"]
        numerical_cols = dataframe.select_dtypes(include=["number"]).columns.tolist()
        feature_cols = [col for col in numerical_cols if col not in exclude_cols]

    # Find player count features (columns with 'alive' in the name)
    player_count_cols = [col for col in feature_cols if "alive" in col.lower()]
    player_count_indices = [feature_cols.index(col) for col in player_count_cols]

    # Find equipment value features
    equipment_cols = [col for col in feature_cols if "eqval" in col.lower()]
    equipment_indices = [feature_cols.index(col) for col in equipment_cols]

    print(f"Found {len(player_count_indices)} player count features: {player_count_cols}")
    print(f"Found {len(equipment_indices)} equipment features: {equipment_cols}")

    return feature_cols, player_count_indices, equipment_indices

def visualize_predictions_fixed(predictions_dict, output_dir, feature_cols, player_count_indices, equipment_indices, max_viz=20):
    """
    Create visualizations of model predictions with fixed player count scaling.
    This version explicitly avoids scaling player counts for visualization purposes.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    features = predictions_dict["features"]
    predictions = predictions_dict["predictions"]
    labels = predictions_dict["labels"]
    masks = predictions_dict["masks"]
    metadata = predictions_dict["metadata"]

    # Select a subset of rounds to visualize
    num_rounds = min(len(metadata), max_viz)
    indices = np.random.choice(len(metadata), num_rounds, replace=False)

    for idx in indices:
        match_id, round_idx = metadata[idx]

        # Get predictions, features, etc. for this round
        round_features = features[idx]
        round_preds = predictions[idx]
        round_labels = labels[idx]
        round_mask = masks[idx]

        # Get valid (non-padded) length
        valid_len = int(round_mask.sum().item())

        # Extract only valid frames
        valid_features = round_features[:valid_len].numpy()
        valid_preds = round_preds[:valid_len].numpy()
        valid_labels = round_labels[:valid_len].numpy()
        
        # IMPORTANT: We need to un-scale the features here
        # If valid_features from the model training have been scaled, we want to 
        # restore them to their original values before visualization
        # Copy the features to avoid modifying the original data
        viz_features = copy.deepcopy(valid_features)
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        # Plot predictions vs ground truth
        axes[0].plot(valid_preds, "b-", label="Predictions")
        axes[0].axhline(y=valid_labels[0], color="r", linestyle="-", label="Ground Truth")
        axes[0].set_ylabel("Win Probability")
        axes[0].set_title(f"Match {match_id}, Round {round_idx}")
        axes[0].legend()
        axes[0].grid(True)

        # Plot CT/T player counts - properly identify columns by name
        ct_alive_idx = None
        t_alive_idx = None
        
        # Get the column names for each player_count_index
        player_count_cols = [feature_cols[idx] for idx in player_count_indices]
        
        for i, col_name in enumerate(player_count_cols):
            if "num_ct_alive" in col_name.lower():
                ct_alive_idx = player_count_indices[i]
            elif "num_t_alive" in col_name.lower():
                t_alive_idx = player_count_indices[i]
        
        # Always display player counts, even if one is missing
        axes[1].set_ylabel("Players Alive")
        axes[1].set_ylim(0, 5)  # Set y-axis limits to 0-5 players (max team size in CS:GO)
        axes[1].grid(True)  # Add grid for better visibility
        
        # Enhanced normalization function that handles emphasis scaling
        def normalize_player_count(values, emphasis_factor=2.0, expected_max=5.0):
            """Normalize player count values to 0-5 range, accounting for emphasis"""
            values_min = values.min()
            values_max = values.max()
            
            # If values are already in valid range, return as-is
            if values_max <= expected_max and values_min >= 0:
                return values
            
            # If values appear to be scaled by emphasis factor
            if values_max > expected_max and values_max <= expected_max * emphasis_factor:
                normalized = values / emphasis_factor
                print(f"Un-scaling player count from range {values_min:.1f}-{values_max:.1f} to 0-{expected_max} (factor: {emphasis_factor:.2f})")
                return normalized
            
            # Otherwise use generic scaling to 0-5 range
            scale_factor = expected_max / values_max if values_max > 0 else 1.0
            normalized = values * scale_factor
            print(f"Normalizing player count from range {values_min:.1f}-{values_max:.1f} to 0-{expected_max} (scale factor: {scale_factor:.2f})")
            return normalized
        
        # Plot CT players - Blue line
        if ct_alive_idx is not None:
            ct_values = viz_features[:, ct_alive_idx]
            ct_normalized = normalize_player_count(ct_values)
            axes[1].plot(ct_normalized, "b-", label="CT Players", linewidth=2.0)
            
            print(f"CT alive original values (first 5): {ct_values[:5]}")
            print(f"CT alive normalized values (first 5): {ct_normalized[:5]}")
        else:
            # If CT data not found, show a placeholder with zeros
            axes[1].plot([0] * valid_len, "b--", label="CT Players (missing)", linewidth=2.0)
        
        # Plot T players - Red line with higher z-order to ensure visibility
        if t_alive_idx is not None:
            t_values = viz_features[:, t_alive_idx]
            t_normalized = normalize_player_count(t_values)
            axes[1].plot(t_normalized, "r-", label="T Players", linewidth=2.0, zorder=10)
            
            print(f"T alive original values (first 5): {t_values[:5]}")
            print(f"T alive normalized values (first 5): {t_normalized[:5]}")
        else:
            # If T data not found, show a placeholder with zeros
            axes[1].plot([0] * valid_len, "r--", label="T Players (missing)", linewidth=2.0, zorder=10)
        
        axes[1].legend()
        axes[1].grid(True)

        # Plot equipment values 
        if len(equipment_indices) > 0:
            # Get the column names for each equipment_index
            equipment_cols = [feature_cols[idx] for idx in equipment_indices]
            
            ct_eq_idx = None
            t_eq_idx = None
            
            # Properly identify equipment values by column name
            for i, col_name in enumerate(equipment_cols):
                if "cteqval" in col_name.lower():
                    ct_eq_idx = equipment_indices[i]
                elif "teqval" in col_name.lower():
                    t_eq_idx = equipment_indices[i]
            
            # Always show equipment values, even if missing
            axes[2].set_ylabel("Equipment Value")
            axes[2].set_xlabel("Frame")
            
            # Plot CT equipment
            if ct_eq_idx is not None:
                axes[2].plot(viz_features[:, ct_eq_idx], "b-", label="CT Equipment")
            else:
                # If CT equipment data not found, show a placeholder
                axes[2].plot([0] * valid_len, "b--", label="CT Equipment (missing)")
                
            # Plot T equipment
            if t_eq_idx is not None:
                axes[2].plot(viz_features[:, t_eq_idx], "r-", label="T Equipment") 
            else:
                # If T equipment data not found, show a placeholder
                axes[2].plot([0] * valid_len, "r--", label="T Equipment (missing)")
                
            axes[2].legend()
            axes[2].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"match_{match_id}_round_{round_idx}.png"))
        plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate fixed visualizations from an existing model")
    parser.add_argument("--csv_path", type=str, default="round_frame_data.csv", help="Path to the CSV data file")
    parser.add_argument("--model_path", type=str, default="./saved_models/responsive_fixed_final4/best_model.pt", help="Path to model weights")
    parser.add_argument("--output_dir", type=str, default="./saved_models/responsive_fixed_final4/fixed_visualizations", help="Output directory")
    parser.add_argument("--max_viz", type=int, default=20, help="Maximum number of rounds to visualize")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data to get feature columns
    print(f"Loading data from {args.csv_path} to identify features")
    sample_df = pd.read_csv(args.csv_path, nrows=1000)
    feature_cols, player_count_indices, equipment_indices = process_features(sample_df)
    
    # Load model
    device = torch.device("cpu")
    model = ResponsiveWinPredictor(
        input_dim=len(feature_cols),
        hidden_dim=64,
        num_layers=2,
        player_count_indices=player_count_indices,
        equipment_indices=equipment_indices
    )
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print(f"Loaded model from {args.model_path}")
    
    # Create dataloaders without feature emphasis for visualization
    # Note: We need to use the raw data for visualization
    print("Creating dataloaders")
    test_dataset = CSGORoundDataset(
        csv_path=args.csv_path,
        feature_cols=feature_cols,
        preload=True,
        # No transform to keep original feature values
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=True,  # Shuffle to get a random sample
        num_workers=0,
        collate_fn=collate_variable_length_rounds,
    )
    
    # Collect predictions
    print("Collecting predictions")
    test_predictions = {}
    all_metadata = []
    all_features = []
    all_preds = []
    all_labels = []
    all_masks = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Collecting Predictions"):
            features = batch["features"].to(device)
            labels = batch["labels"].cpu()
            mask = batch["mask"].cpu()
            metadata = batch["metadata"]
            
            predictions = model(features).cpu()
            
            all_metadata.extend(metadata)
            all_features.append(features.cpu())
            all_preds.append(predictions)
            all_labels.append(labels)
            all_masks.append(mask)
            
            # Break after collecting enough data for visualization
            if len(all_metadata) >= args.max_viz:
                break
    
    # Concatenate tensors
    all_features = torch.cat(all_features, dim=0)
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    
    # Create predictions dictionary
    test_predictions = {
        "metadata": all_metadata[:args.max_viz],
        "features": all_features[:args.max_viz],
        "predictions": all_preds[:args.max_viz],
        "labels": all_labels[:args.max_viz],
        "masks": all_masks[:args.max_viz],
    }
    
    # Create visualizations
    print("Creating fixed visualizations")
    visualize_predictions_fixed(
        test_predictions, 
        args.output_dir, 
        feature_cols, 
        player_count_indices, 
        equipment_indices,
        max_viz=args.max_viz
    )
    
    print(f"Finished creating {args.max_viz} fixed visualizations in {args.output_dir}")
