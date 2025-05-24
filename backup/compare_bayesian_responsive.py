#!/usr/bin/env python3
"""
Bayesian vs Responsive Visualization Comparison
===================================================

This script creates a side-by-side comparison of the Bayesian and responsive
model visualizations for the same match and round.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import argparse
import sys
import os

def compare_visualizations(bayesian_viz_path, responsive_viz_path, output_path):
    """Create a side-by-side comparison of Bayesian and responsive visualizations."""
    
    try:
        # Load both images
        bayesian_img = mpimg.imread(bayesian_viz_path)
        responsive_img = mpimg.imread(responsive_viz_path)
    except Exception as e:
        print(f"Error loading images: {e}")
        return False
    
    # Extract match and round information from the file name
    match_id = os.path.basename(bayesian_viz_path).split("_round_")[0].split("match_")[-1]
    round_idx = os.path.basename(bayesian_viz_path).split("_round_")[-1].split(".png")[0]
    
    # Create figure for side-by-side comparison with more space
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot Bayesian visualization
    axes[0].imshow(bayesian_img)
    axes[0].set_title("Bayesian LSTM Visualization\nWith Uncertainty Quantification", 
                     fontsize=16, fontweight='bold', color='navy')
    axes[0].axis('off')
    
    # Plot responsive visualization
    axes[1].imshow(responsive_img)
    axes[1].set_title("Responsive Model Visualization\nDeterministic Prediction", 
                     fontsize=16, fontweight='bold', color='darkred')
    axes[1].axis('off')
    
    # Add main title
    plt.suptitle(f"CS:GO Win Prediction Comparison for Match {match_id}, Round {round_idx}", 
                fontsize=18, fontweight='bold', y=0.98)
    
    # Add detailed explanation
    plt.figtext(0.5, 0.01, 
                "Comparison of Bayesian and responsive model visualizations.\n\n"
                "The Bayesian model provides uncertainty bounds (confidence intervals) showing the "
                "range of possible predictions, accounting for model uncertainty.\n"
                "This is particularly valuable in situations with limited data or high variability.\n\n"
                "The responsive model gives deterministic point predictions without quantifying uncertainty. "
                "It may be more responsive to sudden events like player eliminations.",
                ha="center", fontsize=12, bbox={"facecolor":"whitesmoke", "alpha":0.9, "pad":10, 
                                               "boxstyle":"round,pad=0.5"},
                wrap=True)
    
    # Add key features comparison as a table at the bottom
    comparison_text = """
    | Feature | Bayesian Model | Responsive Model |
    |---------|---------------|------------------|
    | Uncertainty Quantification | ✓ (±1σ and 95% CI) | ✗ |
    | Event Responsiveness | Medium | High |
    | Prediction Confidence | Varies based on data | Fixed |
    | Training Time | Longer | Shorter |
    | Inference Time | Slower (multiple samples) | Faster |
    """
    
    plt.figtext(0.5, 0.10, comparison_text, ha="center", fontsize=11, 
               family="monospace", bbox={"facecolor":"white", "alpha":0.9, "pad":5})
    
    plt.tight_layout(rect=[0, 0.15, 1, 0.97])
    
    # Save comparison
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Comparison saved to {output_path}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create comparison of Bayesian and responsive visualizations")
    
    parser.add_argument("--bayesian_viz", type=str, 
                       default="./visualizations/bayesian_match/bayesian_match_03e1f233-579c-462d-ac0e-1635d4718ef8.json_round_2.png",
                       help="Path to Bayesian visualization")
    
    parser.add_argument("--responsive_viz", type=str,
                       default="./saved_models/responsive_fixed_final4/visualizations/match_03e1f233-579c-462d-ac0e-1635d4718ef8.json_round_2.png",
                       help="Path to responsive visualization")
    
    parser.add_argument("--output", type=str,
                       default="./bayesian_vs_responsive_comparison.png",
                       help="Output path for comparison image")
    
    args = parser.parse_args()
    
    # Check if files exist
    bayesian_path = Path(args.bayesian_viz)
    responsive_path = Path(args.responsive_viz)
    
    missing_files = []
    if not bayesian_path.exists():
        missing_files.append(str(bayesian_path))
    
    if not responsive_path.exists():
        missing_files.append(str(responsive_path))
    
    if missing_files:
        print(f"Error: The following files were not found:")
        for file_path in missing_files:
            print(f"  - {file_path}")
            
        # Try to find responsive visualization in different directories
        if not responsive_path.exists():
            print("\nAttempting to find responsive visualization in alternative locations...")
            possible_locations = [
                "./saved_models/responsive_fixed_final3/visualizations/",
                "./saved_models/responsive_fixed_final2/visualizations/",
                "./saved_models/responsive_fixed_final/visualizations/",
                "./saved_models/responsive_final/visualizations/",
                "./saved_models/responsive/visualizations/",
                "./visualizations/"
            ]
            
            for location in possible_locations:
                location_path = Path(location)
                if location_path.exists():
                    print(f"Checking {location}...")
                    matches = list(location_path.glob(f"*{responsive_path.name}"))
                    
                    if matches:
                        print(f"Found match: {matches[0]}")
                        responsive_path = matches[0]
                        break
        
        # Exit if files are still missing
        if not bayesian_path.exists() or not responsive_path.exists():
            print("Could not find required visualization files. Please create them first.")
            sys.exit(1)
    
    success = compare_visualizations(bayesian_path, responsive_path, args.output)
    
    if success:
        print("\n✅ Comparison visualization created successfully!")
        print("Key differences:")
        print("1. The Bayesian model includes uncertainty bands (±1σ and 95% confidence intervals)")
        print("2. Both visualizations show player counts normalized to 0-5 range")
        print("3. Both visualizations include equipment values")
        print("4. The ground truth label is included in both visualizations")
    else:
        print("❌ Failed to create comparison visualization.")
