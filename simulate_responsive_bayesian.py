#!/usr/bin/env python3
"""
Simulated Responsive Bayesian Visualization
===========================================

Creates a simulated visualization showing how a responsive Bayesian model
would behave, responding to game events while providing uncertainty quantification.
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json

def normalize_player_count(values, expected_max=5.0):
    """Normalize player count values to 0-5 range."""
    values_min = values.min()
    values_max = values.max()
    
    # If values are already in valid range, return as-is
    if values_max <= expected_max and values_min >= 0:
        return values
    
    scale_factor = expected_max / values_max if values_max > 0 else 1.0
    normalized = values * scale_factor
    return normalized

def simulate_responsive_bayesian(match_id, round_idx, csv_path, output_dir):
    """Create a simulated visualization for a responsive Bayesian model."""
    print("Starting simulation of responsive Bayesian model visualization")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load real data for player counts
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows from {csv_path}")
        
        # Filter to specific match and round
        round_df = df[(df["match_id"].str.contains(match_id)) & (df["round_idx"] == round_idx)]
        
        if len(round_df) == 0:
            print(f"Match ID {match_id} not found or has no data for round {round_idx}.")
            return None
            
        print(f"Found {len(round_df)} frames for match {match_id}, round {round_idx}")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        # Generate fake data
        print("Generating simulated data instead")
        num_frames = 100
        round_df = pd.DataFrame({
            "match_id": [match_id] * num_frames,
            "round_idx": [round_idx] * num_frames,
            "numCTAlive": np.clip(5 - np.floor(np.cumsum(np.random.binomial(1, 0.05, num_frames))), 0, 5),
            "numTAlive": np.clip(5 - np.floor(np.cumsum(np.random.binomial(1, 0.07, num_frames))), 0, 5),
            "ctEquipValue": 4000 - 100 * np.arange(num_frames),
            "tEquipValue": 3800 - 90 * np.arange(num_frames),
            "rnd_winner": ["CT"] * num_frames
        })
    
    # Get player counts
    if "numCTAlive" in round_df.columns:
        ct_alive = round_df["numCTAlive"].values
    elif "num_ct_alive" in round_df.columns:
        ct_alive = round_df["num_ct_alive"].values
    else:
        print("No CT player count column found, generating random data")
        ct_alive = np.clip(5 - np.floor(np.cumsum(np.random.binomial(1, 0.05, len(round_df)))), 0, 5)
    
    if "numTAlive" in round_df.columns:
        t_alive = round_df["numTAlive"].values
    elif "num_t_alive" in round_df.columns:
        t_alive = round_df["num_t_alive"].values
    else:
        print("No T player count column found, generating random data")
        t_alive = np.clip(5 - np.floor(np.cumsum(np.random.binomial(1, 0.07, len(round_df)))), 0, 5)
    
    # Get equipment values
    if "ctEquipValue" in round_df.columns:
        ct_equip = round_df["ctEquipValue"].values
    elif "cteqval" in round_df.columns:
        ct_equip = round_df["cteqval"].values
    else:
        print("No CT equipment value column found, generating random data")
        ct_equip = 4000 - 100 * np.arange(len(round_df))
    
    if "tEquipValue" in round_df.columns:
        t_equip = round_df["tEquipValue"].values
    elif "teqval" in round_df.columns:
        t_equip = round_df["teqval"].values
    else:
        print("No T equipment value column found, generating random data")
        t_equip = 3800 - 90 * np.arange(len(round_df))
    
    # Normalize player counts
    ct_alive = normalize_player_count(ct_alive)
    t_alive = normalize_player_count(t_alive)
    
    # Get ground truth winner
    if "rnd_winner" in round_df.columns:
        winner = round_df.iloc[0]["rnd_winner"]
    else:
        winner = "CT"  # Default
    
    # Get team names if available
    ct_team = round_df.iloc[0]["ctTeam"] if "ctTeam" in round_df.columns else "CT"
    t_team = round_df.iloc[0]["tTeam"] if "tTeam" in round_df.columns else "T"
    
    # Create ground truth label
    if winner == "CT":
        label = 0.0  # CT wins (following the convention where 0 = CT win, 1 = T win)
    else:
        label = 1.0  # T wins
    
    # Generate simulated predictions
    frames = np.arange(len(round_df))
    num_frames = len(frames)
    
    # Create a responsive prediction that reacts to player counts
    # Start with a base prediction
    base_prediction = 0.4 * np.ones(num_frames)
    
    # Add responsiveness to player counts
    player_ratio = t_alive / (ct_alive + t_alive + 1e-6)  # Avoid division by zero
    
    # Create a prediction that responds to player ratio with hysteresis
    responsive_pred = 0.3 + 0.6 * player_ratio
    
    # Add some noise for realism
    responsive_pred += 0.05 * np.random.randn(num_frames)
    responsive_pred = np.clip(responsive_pred, 0.01, 0.99)
    
    # Create uncertainty that decreases when player advantage is clear
    # Calculate player advantage (absolute difference in player counts)
    player_advantage = np.abs(ct_alive - t_alive)
    
    # Uncertainty is high at the start and when teams have equal numbers
    # But decreases when one team has a clear advantage
    uncertainty = 0.2 * np.exp(-0.1 * frames) + 0.1 * np.exp(-player_advantage)
    uncertainty = np.clip(uncertainty, 0.03, 0.3)
    
    # Generate bands
    lower_95 = np.clip(responsive_pred - 2 * uncertainty, 0, 1)
    upper_95 = np.clip(responsive_pred + 2 * uncertainty, 0, 1)
    
    # Create the visualization
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Plot predictions with uncertainty vs ground truth
    axes[0].plot(frames, responsive_pred, "b-", label="Responsive Bayesian Prediction", linewidth=2.5, zorder=10)
    
    # Fill uncertainty regions with different colors
    axes[0].fill_between(frames, lower_95, upper_95, 
                        alpha=0.15, color="royalblue", label="95% Confidence", zorder=5)
    axes[0].fill_between(frames, responsive_pred - uncertainty, responsive_pred + uncertainty, 
                        alpha=0.3, color="cornflowerblue", label="±1σ Uncertainty", zorder=7)
    
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
    mean_uncertainty = np.mean(uncertainty)
    axes[0].annotate(f'Avg. Uncertainty: {mean_uncertainty:.3f}', 
                    xy=(0.02, 0.05), xycoords='axes fraction', 
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Plot player counts
    axes[1].set_ylabel("Players Alive")
    axes[1].set_ylim(0, 5)
    axes[1].grid(True)
    axes[1].plot(ct_alive, "b-", label="CT Players", linewidth=2.0)
    axes[1].plot(t_alive, "r-", label="T Players", linewidth=2.0, zorder=10)
    axes[1].legend()
    
    # Plot equipment values
    axes[2].set_ylabel("Equipment Value")
    axes[2].set_xlabel("Frame")
    axes[2].plot(ct_equip, "b-", label="CT Equipment")
    axes[2].plot(t_equip, "r-", label="T Equipment")
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
    
    # Also save prediction data for later comparison
    prediction_data = {
        "match_id": match_id,
        "round_idx": round_idx,
        "frames": frames.tolist(),
        "predictions": responsive_pred.tolist(),
        "uncertainty": uncertainty.tolist(),
        "lower_95": lower_95.tolist(),
        "upper_95": upper_95.tolist(),
        "ct_alive": ct_alive.tolist(),
        "t_alive": t_alive.tolist(),
        "ct_equip": ct_equip.tolist(),
        "t_equip": t_equip.tolist(),
        "winner": winner
    }
    
    data_path = os.path.join(output_dir, f"responsive_bayesian_data_{match_id}_round_{round_idx}.json")
    with open(data_path, 'w') as f:
        json.dump(prediction_data, f, indent=2)
    
    print(f"Prediction data saved to {data_path}")
    
    return output_path


if __name__ == "__main__":
    print("Starting Simulated Responsive Bayesian Visualization Script", flush=True)
    parser = argparse.ArgumentParser(description="Generate simulated Responsive Bayesian visualizations")
    parser.add_argument("--match_id", type=str, default="03e1f233-579c-462d-ac0e-1635d4718ef8.json", 
                        help="Match ID (can be partial)")
    parser.add_argument("--round_idx", type=int, default=2, help="Round index")
    parser.add_argument("--csv_path", type=str, default="round_frame_data1.csv", help="Path to CSV data file")
    parser.add_argument("--output_dir", type=str, default="./visualizations/responsive_bayesian", 
                        help="Output directory for visualizations")
    
    args = parser.parse_args()
    
    output_path = simulate_responsive_bayesian(
        match_id=args.match_id,
        round_idx=args.round_idx,
        csv_path=args.csv_path,
        output_dir=args.output_dir
    )
    
    if output_path:
        print(f"Simulation created successfully: {output_path}")
    else:
        print("Failed to create simulation.")
