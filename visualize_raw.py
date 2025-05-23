#!/usr/bin/env python3
"""
Script to generate fixed visualizations with non-emphasized player counts.
This creates visualizations that explicitly don't apply the FeatureEmphasis 
scaling to player counts, making them display in their true 0-5 range.
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Import from the project files
from src.mbml.enhanced_model import ResponsiveWinPredictor

def process_features(dataframe, feature_cols=None):
    """Extract features and identify special feature indices."""
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

def visualize_match_round(match_id, round_idx, csv_path, model_path, output_dir):
    """Create visualization for a specific match round without FeatureEmphasis."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the data
    df = pd.read_csv(csv_path)
    
    # Get the feature columns
    feature_cols, player_count_indices, equipment_indices = process_features(df)
    
    # Debug: print unique match IDs
    print(f"Available match IDs (first 5):")
    print(df["match_id"].unique()[:5])
    
    # Extract the specific round
    round_df = df[(df["match_id"] == match_id) & (df["round_idx"] == round_idx)]
    if len(round_df) == 0:
        print(f"Error: No data found for match {match_id}, round {round_idx}")
        print(f"Looking for partial match...")
        # Try a partial match on the match_id
        for avail_match in df["match_id"].unique():
            if match_id in avail_match:
                print(f"Found potential match: {avail_match}")
                round_df = df[(df["match_id"] == avail_match) & (df["round_idx"] == round_idx)]
                if len(round_df) > 0:
                    print(f"Using match_id: {avail_match}")
                    match_id = avail_match
                    break
        
        if len(round_df) == 0:
            return
    
    # Extract features - don't apply any transformations
    features = torch.tensor(round_df[feature_cols].values, dtype=torch.float32)
    
    # Load the model
    model = ResponsiveWinPredictor(
        input_dim=len(feature_cols),
        hidden_dim=64,
        num_layers=2,
        player_count_indices=player_count_indices,
        equipment_indices=equipment_indices
    )
    
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    # Get predictions
    with torch.no_grad():
        predictions = model(features.unsqueeze(0)).squeeze(0)
    
    # Get ground truth
    winner = round_df.iloc[0]["rnd_winner"]
    ct_team = round_df.iloc[0]["ctTeam"]
    t_team = round_df.iloc[0]["tTeam"]
    
    # Create ground truth label
    if winner == "CT":
        label = 1.0  # CT wins
    else:
        label = 0.0  # T wins
    
    # Create the visualization
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Plot predictions vs ground truth
    axes[0].plot(predictions.numpy(), "b-", label="Predictions")
    axes[0].axhline(y=label, color="r", linestyle="-", label="Ground Truth")
    axes[0].set_ylabel("Win Probability")
    axes[0].set_title(f"Match {match_id}, Round {round_idx} - {ct_team} (CT) vs {t_team} (T)")
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot player counts without any transformations
    ct_alive_idx = None
    t_alive_idx = None
    
    # Find the player count columns
    for i, col in enumerate(feature_cols):
        if "num_ct_alive" in col.lower():
            ct_alive_idx = i
        elif "num_t_alive" in col.lower():
            t_alive_idx = i
    
    # Set up the player count plot
    axes[1].set_ylabel("Players Alive")
    axes[1].set_ylim(0, 5)
    axes[1].grid(True)
    
    # Plot CT players
    if ct_alive_idx is not None:
        ct_values = features[:, ct_alive_idx].numpy()
        axes[1].plot(ct_values, "b-", label="CT Players", linewidth=2.0)
        print(f"CT alive values (first 5): {ct_values[:5]}")
        print(f"CT alive values (range): {ct_values.min()} to {ct_values.max()}")
    else:
        axes[1].plot([0] * len(features), "b--", label="CT Players (missing)", linewidth=2.0)
    
    # Plot T players
    if t_alive_idx is not None:
        t_values = features[:, t_alive_idx].numpy()
        axes[1].plot(t_values, "r-", label="T Players", linewidth=2.0, zorder=10)
        print(f"T alive values (first 5): {t_values[:5]}")
        print(f"T alive values (range): {t_values.min()} to {t_values.max()}")
    else:
        axes[1].plot([0] * len(features), "r--", label="T Players (missing)", linewidth=2.0, zorder=10)
    
    axes[1].legend()
    
    # Plot equipment values
    ct_eq_idx = None
    t_eq_idx = None
    
    # Find the equipment value columns
    for i, col in enumerate(feature_cols):
        if "cteqval" in col.lower():
            ct_eq_idx = i
        elif "teqval" in col.lower():
            t_eq_idx = i
    
    axes[2].set_ylabel("Equipment Value")
    axes[2].set_xlabel("Frame")
    
    # Plot CT equipment
    if ct_eq_idx is not None:
        axes[2].plot(features[:, ct_eq_idx].numpy(), "b-", label="CT Equipment")
    else:
        axes[2].plot([0] * len(features), "b--", label="CT Equipment (missing)")
    
    # Plot T equipment
    if t_eq_idx is not None:
        axes[2].plot(features[:, t_eq_idx].numpy(), "r-", label="T Equipment")
    else:
        axes[2].plot([0] * len(features), "r--", label="T Equipment (missing)")
    
    axes[2].legend()
    axes[2].grid(True)
    
    # Save the figure
    plt.tight_layout()
    output_path = output_dir / f"{match_id}_round_{round_idx}_raw.png"
    plt.savefig(output_path)
    plt.close(fig)
    
    print(f"Visualization saved to {output_path}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate fixed visualizations without FeatureEmphasis scaling")
    parser.add_argument("--match_id", type=str, default="03e1f233-579c-462d-ac0e-1635d4718ef8.json", help="Match ID")
    parser.add_argument("--round_idx", type=int, default=1, help="Round index")
    parser.add_argument("--csv_path", type=str, default="round_frame_data.csv", help="Path to CSV data file")
    parser.add_argument("--model_path", type=str, default="./saved_models/responsive_fixed_final4/best_model.pt", help="Path to model weights")
    parser.add_argument("--output_dir", type=str, default="./visualizations_raw", help="Output directory for visualizations")
    
    args = parser.parse_args()
    
    output_path = visualize_match_round(
        match_id=args.match_id,
        round_idx=args.round_idx,
        csv_path=args.csv_path,
        model_path=args.model_path,
        output_dir=args.output_dir
    )
    
    if output_path:
        print(f"Visualization created successfully: {output_path}")
