#!/usr/bin/env python3
"""
Responsive Bayesian Visualization
=================================

Creates visualizations for the Responsive Bayesian LSTM model, showing:
1. Predictions with uncertainty bands
2. Player counts
3. Equipment values
4. Clear responsiveness to game events
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Import from the project files
import pyro
from responsive_bayesian_lstm import ResponsiveBayesianLSTM


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


def predict_with_uncertainty(model, features, num_samples=30):
    """Get predictions with uncertainty estimates using Monte Carlo sampling."""
    model.train()  # Enable dropout for sampling
    predictions = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            pred = model(features.unsqueeze(0))
            predictions.append(pred.squeeze(0).cpu().numpy())
    
    predictions = np.array(predictions)
    
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)
    lower_95 = np.percentile(predictions, 2.5, axis=0)
    upper_95 = np.percentile(predictions, 97.5, axis=0)
    
    return mean_pred, std_pred, lower_95, upper_95


def normalize_player_count(values, expected_max=5.0):
    """Normalize player count values to 0-5 range, accounting for possible emphasis."""
    values_min = values.min()
    values_max = values.max()
    
    # If values are already in valid range, return as-is
    if values_max <= expected_max and values_min >= 0:
        return values
    
    # If values appear to be scaled by emphasis factor
    if values_max > expected_max and values_max <= expected_max * 2:
        normalized = values / 2.0
        print(f"Un-scaling player count from range {values_min:.1f}-{values_max:.1f} to 0-{expected_max}")
        return normalized
    
    # Otherwise use generic scaling to 0-5 range
    scale_factor = expected_max / values_max if values_max > 0 else 1.0
    normalized = values * scale_factor
    print(f"Normalizing player count from range {values_min:.1f}-{values_max:.1f} to 0-{expected_max}")
    return normalized


def visualize_match_round(match_id, round_idx, csv_path, model_path, output_dir, num_samples=30):
    """Create visualization for a specific match round with responsive Bayesian uncertainty."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading data from {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded data with {len(df)} rows")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None
    
    # Print first few available match IDs
    print("Available match IDs (first 5):")
    for mid in df["match_id"].unique()[:5]:
        print(f"  - {mid}")
    
    # Handle case where match_id might be partial or exact
    if match_id is not None:
        round_df = df[(df["match_id"].str.contains(match_id)) & (df["round_idx"] == round_idx)]
        if len(round_df) == 0:
            print(f"Match ID {match_id} not found or has no data for round {round_idx}.")
            print("Trying to find a valid match...")
            available_matches = df["match_id"].unique()
            
            for avail_match in available_matches:
                print(f"Found potential match: {avail_match}")
                round_df = df[(df["match_id"] == avail_match) & (df["round_idx"] == round_idx)]
                if len(round_df) > 0:
                    print(f"Using match_id: {avail_match}")
                    match_id = avail_match
                    break
        
        if len(round_df) == 0:
            print("No suitable data found.")
            return None
    else:
        # If no match_id provided, use the first match with data for the specified round
        round_df = df[df["round_idx"] == round_idx]
        if len(round_df) == 0:
            print(f"No data found for round {round_idx}.")
            return None
        match_id = round_df.iloc[0]["match_id"]
        print(f"Using first available match: {match_id}")

    # Process features
    feature_cols, player_count_indices, equipment_indices = process_features(round_df)
    
    # Extract features
    features = torch.tensor(round_df[feature_cols].values, dtype=torch.float32)
    
    # Load the Responsive Bayesian model
    print(f"Loading Responsive Bayesian model from {model_path}")
    try:
        # Use weights_only=False to allow loading Pyro distribution objects
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        print("Successfully loaded checkpoint")
        
        model = ResponsiveBayesianLSTM(
            input_dim=checkpoint['input_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            num_layers=checkpoint['num_layers'],
            dropout=checkpoint['dropout'],
            player_count_indices=checkpoint.get('player_count_indices', player_count_indices),
            equipment_indices=checkpoint.get('equipment_indices', equipment_indices),
            use_entmax=checkpoint.get('use_entmax', True)
        )
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        print(f"Error loading model: {e}")
        # Fallback: Try to create a new model with default parameters
        print("Attempting to create a model with default parameters...")
        model = ResponsiveBayesianLSTM(
            input_dim=len(feature_cols),
            hidden_dim=64,
            num_layers=2,
            dropout=0.2,
            player_count_indices=player_count_indices,
            equipment_indices=equipment_indices
        )
        return None
    
    # Handle feature dimension mismatch
    expected_dim = checkpoint['input_dim']
    actual_dim = features.shape[1]
    
    if actual_dim != expected_dim:
        print(f"Feature dimension mismatch: model expects {expected_dim}, data has {actual_dim}")
        
        # Pad or truncate features to match model's expected input size
        if actual_dim < expected_dim:
            print(f"Padding features from {actual_dim} to {expected_dim} dimensions")
            padding = torch.zeros((features.shape[0], expected_dim - actual_dim), dtype=torch.float32)
            features = torch.cat([features, padding], dim=1)
        elif actual_dim > expected_dim:
            print(f"Truncating features from {actual_dim} to {expected_dim} dimensions")
            features = features[:, :expected_dim]
    
    # Get predictions with uncertainty
    print(f"Generating {num_samples} prediction samples for uncertainty estimation...")
    mean_pred, std_pred, lower_95, upper_95 = predict_with_uncertainty(model, features, num_samples)
    
    # Get ground truth
    try:
        winner = round_df.iloc[0]["rnd_winner"]
        ct_team = round_df.iloc[0]["ctTeam"] if "ctTeam" in round_df.columns else "CT"
        t_team = round_df.iloc[0]["tTeam"] if "tTeam" in round_df.columns else "T"
    except Exception as e:
        print(f"Error getting round metadata: {e}")
        print(f"Available columns: {round_df.columns.tolist()}")
        winner = "CT"  # Default
        ct_team = "CT"
        t_team = "T"
    
    # Create ground truth label
    if winner == "CT":
        label = 0.0  # CT wins (following the convention where 0 = CT win, 1 = T win)
    else:
        label = 1.0  # T wins
    
    # Create the visualization
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    frames = np.arange(len(mean_pred))
    
    # Plot predictions with uncertainty vs ground truth
    axes[0].plot(frames, mean_pred, "b-", label="Responsive Bayesian Prediction", linewidth=2.5, zorder=10)
    
    # Fill uncertainty regions with different colors for better visual distinction
    axes[0].fill_between(frames, lower_95, upper_95, 
                        alpha=0.15, color="royalblue", label="95% Confidence", zorder=5)
    axes[0].fill_between(frames, mean_pred - std_pred, mean_pred + std_pred, 
                        alpha=0.3, color="cornflowerblue", label="±1σ Uncertainty", zorder=7)
    
    # Add visual indicators for highest uncertainty regions
    uncertainty_magnitude = upper_95 - lower_95
    high_uncertainty_threshold = np.percentile(uncertainty_magnitude, 75)
    high_uncertainty_indices = np.where(uncertainty_magnitude > high_uncertainty_threshold)[0]
    
    if len(high_uncertainty_indices) > 0:
        # Mark regions of high uncertainty with vertical spans
        for i in range(len(high_uncertainty_indices)):
            if i == 0 or high_uncertainty_indices[i] > high_uncertainty_indices[i-1] + 5:
                axes[0].axvspan(high_uncertainty_indices[i] - 2, high_uncertainty_indices[i] + 2, 
                               alpha=0.1, color='red', zorder=1)
    
    # Add ground truth
    axes[0].axhline(y=label, color="r", linestyle="-", 
                   label=f"Ground Truth ({winner} win)", linewidth=1.5, zorder=8)
    
    # Add horizontal lines at 0.25, 0.5, and 0.75 for reference
    axes[0].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, zorder=2)
    axes[0].axhline(y=0.25, color="gray", linestyle=":", alpha=0.4, zorder=2)
    axes[0].axhline(y=0.75, color="gray", linestyle=":", alpha=0.4, zorder=2)
    
    # Enhanced title and styling
    axes[0].set_title(f"Match {match_id}, Round {round_idx} - {ct_team} (CT) vs {t_team} (T)", 
                     fontsize=12, fontweight='bold')
    axes[0].set_ylabel("T Win Probability", fontweight='bold')
    axes[0].legend(loc='upper right', framealpha=0.9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-0.05, 1.05)
    
    # Add uncertainty measure annotation
    mean_uncertainty = np.mean(uncertainty_magnitude)
    axes[0].annotate(f'Avg. Uncertainty: {mean_uncertainty:.3f}', 
                    xy=(0.02, 0.05), xycoords='axes fraction', 
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Plot player counts
    ct_alive_idx = None
    t_alive_idx = None
    
    # Find the player count columns
    for i, col in enumerate(feature_cols):
        if "num_ct_alive" in col.lower() or "numctalive" in col.lower():
            ct_alive_idx = i
        elif "num_t_alive" in col.lower() or "numtalive" in col.lower():
            t_alive_idx = i
    
    # Set up the player count plot
    axes[1].set_ylabel("Players Alive")
    axes[1].set_ylim(0, 5)
    axes[1].grid(True)
    
    # Plot CT players
    if ct_alive_idx is not None:
        ct_values = features[:, ct_alive_idx].numpy()
        ct_normalized = normalize_player_count(ct_values)
        axes[1].plot(ct_normalized, "b-", label="CT Players", linewidth=2.0)
        print(f"CT alive values (first 5): {ct_normalized[:5]}")
    else:
        axes[1].plot([0] * len(features), "b--", label="CT Players (missing)", linewidth=2.0)
    
    # Plot T players
    if t_alive_idx is not None:
        t_values = features[:, t_alive_idx].numpy()
        t_normalized = normalize_player_count(t_values)
        axes[1].plot(t_normalized, "r-", label="T Players", linewidth=2.0, zorder=10)
        print(f"T alive values (first 5): {t_normalized[:5]}")
    else:
        axes[1].plot([0] * len(features), "r--", label="T Players (missing)", linewidth=2.0, zorder=10)
    
    axes[1].legend()
    
    # Plot equipment values
    ct_eq_idx = None
    t_eq_idx = None
    
    # Find the equipment value columns
    for i, col in enumerate(feature_cols):
        if "cteqval" in col.lower() or "ctequipvalue" in col.lower():
            ct_eq_idx = i
        elif "teqval" in col.lower() or "tequipvalue" in col.lower():
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
    
    # Add subtitle highlighting responsive nature
    plt.figtext(0.5, 0.01, 
               "Responsive Bayesian LSTM: Combines uncertainty quantification with event responsiveness.\n"
               "Notice how the prediction changes when player counts change, unlike the basic Bayesian model.",
               ha="center", fontsize=10, bbox=dict(facecolor="whitesmoke", alpha=0.8, pad=5))
    
    # Save the figure
    output_path = os.path.join(output_dir, f"responsive_bayesian_match_{match_id}_round_{round_idx}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Visualization saved to {output_path}")
    return output_path


if __name__ == "__main__":
    print("Starting Responsive Bayesian Visualization Script")
    parser = argparse.ArgumentParser(description="Generate Responsive Bayesian visualizations for CS:GO matches")
    parser.add_argument("--match_id", type=str, default="03e1f233-579c-462d-ac0e-1635d4718ef8.json", 
                        help="Match ID (can be partial)")
    parser.add_argument("--round_idx", type=int, default=2, help="Round index")
    parser.add_argument("--csv_path", type=str, default="round_frame_data1.csv", help="Path to CSV data file")
    parser.add_argument("--model_path", type=str, default="./saved_models/responsive_bayesian/best_model.pt", 
                        help="Path to model weights")
    parser.add_argument("--output_dir", type=str, default="./saved_models/responsive_bayesian", 
                        help="Output directory for visualizations")
    parser.add_argument("--num_samples", type=int, default=30, 
                        help="Number of MC samples for uncertainty estimation")
    
    args = parser.parse_args()
    
    output_path = visualize_match_round(
        match_id=args.match_id,
        round_idx=args.round_idx,
        csv_path=args.csv_path,
        model_path=args.model_path,
        output_dir=args.output_dir,
        num_samples=args.num_samples
    )
    
    if output_path:
        print(f"Visualization created successfully: {output_path}")
    else:
        print("Failed to create visualization.")
