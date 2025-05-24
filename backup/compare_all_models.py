#!/usr/bin/env python3
"""
Three-Way Model Comparison
===========================

Creates a side-by-side comparison of all three models:
1. Original Bayesian LSTM (with uncertainty but not responsive)
2. Responsive LSTM (responsive but without uncertainty)
3. Responsive Bayesian LSTM (our new hybrid with both features)
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import sys

def compare_all_models(bayesian_viz_path, responsive_viz_path, responsive_bayesian_viz_path, output_path):
    """Create a side-by-side comparison of all three model visualizations."""
    
    try:
        # Load all three images
        bayesian_img = mpimg.imread(bayesian_viz_path)
        responsive_img = mpimg.imread(responsive_viz_path)
        responsive_bayesian_img = mpimg.imread(responsive_bayesian_viz_path)
    except Exception as e:
        print(f"Error loading images: {e}")
        return False
    
    # Extract match and round information from the file name
    match_id = os.path.basename(bayesian_viz_path).split("_round_")[0].split("match_")[-1]
    round_idx = os.path.basename(bayesian_viz_path).split("_round_")[-1].split(".png")[0]
    
    # Create figure for side-by-side comparison with more space
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # Plot Bayesian visualization
    axes[0].imshow(bayesian_img)
    axes[0].set_title("Bayesian LSTM Model\nUncertainty but Low Responsiveness", 
                     fontsize=14, fontweight='bold', color='navy')
    axes[0].axis('off')
    
    # Plot responsive visualization
    axes[1].imshow(responsive_img)
    axes[1].set_title("Responsive Model\nHigh Responsiveness but No Uncertainty", 
                     fontsize=14, fontweight='bold', color='darkred')
    axes[1].axis('off')
    
    # Plot responsive Bayesian visualization
    axes[2].imshow(responsive_bayesian_img)
    axes[2].set_title("Responsive Bayesian LSTM\nCombines Uncertainty and Responsiveness", 
                     fontsize=14, fontweight='bold', color='darkgreen')
    axes[2].axis('off')
    
    # Add main title
    plt.suptitle(f"CS:GO Win Prediction Model Comparison - Match {match_id}, Round {round_idx}", 
                fontsize=18, fontweight='bold', y=0.98)
    
    # Add detailed explanation
    plt.figtext(0.5, 0.02, 
                "Three-way comparison of win prediction models:\n\n"
                "1. Bayesian LSTM: Provides uncertainty quantification but has limited responsiveness to game events.\n"
                "2. Responsive Model: Highly responsive to game events but lacks uncertainty quantification.\n"
                "3. Responsive Bayesian LSTM: Our new hybrid model that combines uncertainty quantification with "
                "responsiveness to critical game events like player eliminations.",
                ha="center", fontsize=12, bbox={"facecolor":"whitesmoke", "alpha":0.9, "pad":10, 
                                               "boxstyle":"round,pad=0.5"},
                wrap=True)
    
    # Add feature comparison table
    comparison_text = """
    | Feature | Bayesian LSTM | Responsive Model | Responsive Bayesian |
    |---------|---------------|------------------|---------------------|
    | Uncertainty Quantification | ✓ | ✗ | ✓ |
    | Event Responsiveness | Low | High | High |
    | Confidence Calibration | Good | Poor | Good |
    | Player Count Sensitivity | Low | High | High |
    | Equipment Value Usage | Basic | Advanced | Advanced |
    | Computational Cost | Medium | Low | High |
    """
    
    plt.figtext(0.5, 0.12, comparison_text, ha="center", fontsize=11, 
               family="monospace", bbox={"facecolor":"white", "alpha":0.9, "pad":5})
    
    plt.tight_layout(rect=[0, 0.22, 1, 0.97])
    
    # Save comparison
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Comparison saved to {output_path}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create comparison of all three model visualizations")
    
    parser.add_argument("--bayesian_viz", type=str, 
                       default="./visualizations/bayesian_match/bayesian_match_03e1f233-579c-462d-ac0e-1635d4718ef8.json_round_2.png",
                       help="Path to Bayesian visualization")
    
    parser.add_argument("--responsive_viz", type=str,
                       default="./saved_models/responsive/visualizations/match_03e1f233-579c-462d-ac0e-1635d4718ef8.json_round_2.png",
                       help="Path to responsive visualization")
    
    parser.add_argument("--responsive_bayesian_viz", type=str,
                       default="./visualizations/responsive_bayesian/responsive_bayesian_match_03e1f233-579c-462d-ac0e-1635d4718ef8.json_round_2.png",
                       help="Path to responsive Bayesian visualization")
    
    parser.add_argument("--output", type=str,
                       default="./three_model_comparison.png",
                       help="Output path for comparison image")
    
    args = parser.parse_args()
    
    # Check if files exist
    bayesian_path = Path(args.bayesian_viz)
    responsive_path = Path(args.responsive_viz)
    responsive_bayesian_path = Path(args.responsive_bayesian_viz)
    
    missing_files = []
    if not bayesian_path.exists():
        missing_files.append(str(bayesian_path))
    
    if not responsive_path.exists():
        missing_files.append(str(responsive_path))
        
    if not responsive_bayesian_path.exists():
        missing_files.append(str(responsive_bayesian_path))
    
    if missing_files:
        print(f"Error: The following files were not found:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        
        print("\nPlease create the missing visualizations first.")
        print("For the responsive Bayesian model, run:")
        print("python train_responsive_bayesian.py")
        print("python responsive_bayesian_viz.py")
        sys.exit(1)
    
    success = compare_all_models(
        bayesian_path, 
        responsive_path, 
        responsive_bayesian_path,
        args.output
    )
    
    if success:
        print("\n✅ Three-way comparison visualization created successfully!")
        print("Key differences:")
        print("1. The Bayesian LSTM provides uncertainty bands but has low responsiveness to game events")
        print("2. The Responsive model reacts quickly to game events but lacks uncertainty quantification")
        print("3. The Responsive Bayesian LSTM combines the strengths of both approaches")
    else:
        print("❌ Failed to create comparison visualization.")
