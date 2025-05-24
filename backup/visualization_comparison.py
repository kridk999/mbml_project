#!/usr/bin/env python3
"""
Visualization comparison tool for CS:GO player count scaling issues.
Shows three visualizations side by side to highlight the issue and solution:
1. Original visualization with doubled player count scaling (0-10)
2. Visualization with FeatureEmphasis disabled (raw values 0-5) 
3. Visualization with normalized values (scaling back to 0-5)
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import sys

# Fake data to demonstrate the visualization issue
def generate_sample_data():
    """Generate sample data to simulate the player count visualization issue."""
    # Sequence length
    seq_len = 100
    
    # Create some initial player count values (5 players at the start)
    ct_raw = np.ones(seq_len) * 5
    t_raw = np.ones(seq_len) * 5
    
    # Make player counts decrease over time to simulate kills
    # CT player count changes: 5 → 4 → 3
    ct_raw[30:60] = 4
    ct_raw[60:] = 3
    
    # T player count changes: 5 → 4 → 3 → 2 → 1 → 0
    t_raw[20:40] = 4
    t_raw[40:60] = 3 
    t_raw[60:80] = 2
    t_raw[80:90] = 1
    t_raw[90:] = 0
    
    # Create "emphasized" version that doubles the player counts
    ct_emphasized = ct_raw * 2
    t_emphasized = t_raw * 2
    
    # Create ground truth labels
    # If T reaches 0, CT wins
    labels = np.zeros(seq_len)
    if 0 in t_raw:
        labels[:] = 1.0  # CT wins
    
    # Create prediction values that react to player count changes
    predictions = np.zeros(seq_len)
    predictions[:20] = 0.2  # Starting prediction favors T
    predictions[20:40] = 0.3  # Slight shift when first T dies
    predictions[40:60] = 0.45  # Moving toward 50/50
    predictions[60:80] = 0.6  # CT now favored
    predictions[80:90] = 0.8  # Strong CT favor
    predictions[90:] = 0.95  # Almost certain CT win
    
    return {
        "ct_raw": ct_raw,
        "t_raw": t_raw,
        "ct_emphasized": ct_emphasized,
        "t_emphasized": t_emphasized,
        "labels": labels,
        "predictions": predictions,
        "seq_len": seq_len
    }

def create_comparison_visualization(output_path=None):
    """Create a visual comparison of the three visualization approaches."""
    data = generate_sample_data()
    
    # Create a figure with three subplots side by side
    fig, axes = plt.subplots(3, 3, figsize=(18, 10))
    fig.suptitle("CS:GO Win Prediction: Player Count Visualization Comparison", fontsize=16)
    
    # Common x values
    x = np.arange(data["seq_len"])
    
    # Set column titles
    axes[0, 0].set_title("Original (Scaled 0-10)", fontweight='bold')
    axes[0, 1].set_title("Raw Values (0-5)", fontweight='bold')
    axes[0, 2].set_title("Normalized (Re-scaled to 0-5)", fontweight='bold')
    
    # --------------------------- #
    # Column 1: Original - Scaled Values (0-10)
    # --------------------------- #
    
    # Win probability plot
    axes[0, 0].plot(data["predictions"], "b-", label="Predictions")
    axes[0, 0].axhline(y=data["labels"][0], color="r", linestyle="-", label="Ground Truth")
    axes[0, 0].set_ylabel("Win Probability")
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    axes[0, 0].set_ylim(0, 1)
    
    # Player count plot - SCALED values
    axes[1, 0].plot(data["ct_emphasized"], "b-", label="CT Players", linewidth=2.0)
    axes[1, 0].plot(data["t_emphasized"], "r-", label="T Players", linewidth=2.0, zorder=10)
    axes[1, 0].set_ylabel("Players Alive")
    axes[1, 0].set_ylim(0, 10)  # Y-axis from 0-10 (scaled)
    axes[1, 0].grid(True)
    axes[1, 0].legend()
    
    # Equipment value plot
    eq_values_ct = 1000 + np.random.rand(data["seq_len"]) * 2000
    eq_values_t = 1000 + np.random.rand(data["seq_len"]) * 2000
    axes[2, 0].plot(eq_values_ct, "b-", label="CT Equipment")
    axes[2, 0].plot(eq_values_t, "r-", label="T Equipment")
    axes[2, 0].set_ylabel("Equipment Value")
    axes[2, 0].set_xlabel("Frame")
    axes[2, 0].grid(True)
    axes[2, 0].legend()
    
    # Add problem annotation
    axes[1, 0].annotate("PROBLEM: Values scaled to 0-10\ninstead of 0-5 (actual team sizes)",
                     xy=(data["seq_len"]/2, 9),
                     xytext=(data["seq_len"]/2, 8),
                     bbox=dict(boxstyle="round,pad=0.5", facecolor="red", alpha=0.3),
                     ha='center')
    
    # --------------------------- #
    # Column 2: Raw Values (0-5)
    # --------------------------- #
    
    # Win probability plot
    axes[0, 1].plot(data["predictions"], "b-", label="Predictions")
    axes[0, 1].axhline(y=data["labels"][0], color="r", linestyle="-", label="Ground Truth")
    axes[0, 1].set_ylabel("Win Probability")
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    axes[0, 1].set_ylim(0, 1)
    
    # Player count plot - RAW values
    axes[1, 1].plot(data["ct_raw"], "b-", label="CT Players", linewidth=2.0)
    axes[1, 1].plot(data["t_raw"], "r-", label="T Players", linewidth=2.0, zorder=10)
    axes[1, 1].set_ylabel("Players Alive")
    axes[1, 1].set_ylim(0, 5)  # Y-axis from 0-5 (raw)
    axes[1, 1].grid(True)
    axes[1, 1].legend()
    
    # Equipment value plot
    axes[2, 1].plot(eq_values_ct, "b-", label="CT Equipment")
    axes[2, 1].plot(eq_values_t, "r-", label="T Equipment")
    axes[2, 1].set_ylabel("Equipment Value")
    axes[2, 1].set_xlabel("Frame")
    axes[2, 1].grid(True)
    axes[2, 1].legend()
    
    # Add solution annotation
    axes[1, 1].annotate("SOLUTION #1: Don't use FeatureEmphasis\nfor visualization",
                     xy=(data["seq_len"]/2, 4.5),
                     xytext=(data["seq_len"]/2, 3.7),
                     bbox=dict(boxstyle="round,pad=0.5", facecolor="green", alpha=0.3),
                     ha='center')
    
    # --------------------------- #
    # Column 3: Normalized Values (re-scaled 0-5)
    # --------------------------- #
    
    # Win probability plot
    axes[0, 2].plot(data["predictions"], "b-", label="Predictions")
    axes[0, 2].axhline(y=data["labels"][0], color="r", linestyle="-", label="Ground Truth")
    axes[0, 2].set_ylabel("Win Probability")
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    axes[0, 2].set_ylim(0, 1)
    
    # Player count plot - NORMALIZED values (scale back to 0-5)
    normalized_ct = data["ct_emphasized"] / 2
    normalized_t = data["t_emphasized"] / 2
    axes[1, 2].plot(normalized_ct, "b-", label="CT Players", linewidth=2.0)
    axes[1, 2].plot(normalized_t, "r-", label="T Players", linewidth=2.0, zorder=10)
    axes[1, 2].set_ylabel("Players Alive")
    axes[1, 2].set_ylim(0, 5)  # Y-axis from 0-5 (normalized)
    axes[1, 2].grid(True)
    axes[1, 2].legend()
    
    # Equipment value plot
    axes[2, 2].plot(eq_values_ct, "b-", label="CT Equipment")
    axes[2, 2].plot(eq_values_t, "r-", label="T Equipment")
    axes[2, 2].set_ylabel("Equipment Value")
    axes[2, 2].set_xlabel("Frame")
    axes[2, 2].grid(True)
    axes[2, 2].legend()
    
    # Add solution annotation
    axes[1, 2].annotate("SOLUTION #2: Normalize emphasized values\nby dividing by 2 during visualization",
                     xy=(data["seq_len"]/2, 4.5),
                     xytext=(data["seq_len"]/2, 3.7),
                     bbox=dict(boxstyle="round,pad=0.5", facecolor="green", alpha=0.3),
                     ha='center')
    
    # Show interesting event periods
    for ax in axes[1, :]:
        ax.axvspan(20, 40, alpha=0.2, color='gray')
        ax.axvspan(60, 80, alpha=0.2, color='gray')
        ax.axvspan(90, 100, alpha=0.2, color='gray')
    
    # Add explanatory text at the bottom
    plt.figtext(0.5, 0.01, 
                "CS:GO player counts range from 0-5 players per team. The FeatureEmphasis transformation "
                "doubles these values for model training (0-10 scale).\n"
                "This causes visualization issues that can be fixed by either displaying raw data or "
                "normalizing the values back to the expected 0-5 range.",
                ha="center", fontsize=12, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save or display
    if output_path:
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()
    
    return fig

if __name__ == "__main__":
    # Create the visualization and save to file
    output_path = "fixed_visualization.png"
    if len(sys.argv) > 1:
        output_path = sys.argv[1]
    
    create_comparison_visualization(output_path)
