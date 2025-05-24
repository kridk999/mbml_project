#!/usr/bin/env python3
"""
Tool to compare player count visualization results from different runs.
This script displays visualizations side by side to show the improvement in player count visualization.
"""
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

def compare_visualizations(old_viz_path, new_viz_path):
    """
    Display two visualizations side by side for comparison.
    
    Args:
        old_viz_path: Path to the visualization from the original model
        new_viz_path: Path to the visualization with the fix applied
    """
    # Load the images
    old_img = mpimg.imread(old_viz_path)
    new_img = mpimg.imread(new_viz_path)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot the old visualization
    axes[0].imshow(old_img)
    axes[0].set_title("Original Visualization\nPlayer counts scaled 0-10", fontsize=14)
    axes[0].axis('off')
    
    # Plot the new visualization
    axes[1].imshow(new_img)
    axes[1].set_title("Fixed Visualization\nPlayer counts normalized to 0-5", fontsize=14)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Define default paths
    old_dir = Path("/Users/kristofferkjaer/Desktop/mbml_project/saved_models/responsive_fixed_final3/visualizations")
    new_dir = Path("/Users/kristofferkjaer/Desktop/mbml_project/saved_models/responsive_final_fixed/visualizations")
    
    # Select which round to compare
    round_idx = 1
    if len(sys.argv) > 1:
        round_idx = int(sys.argv[1])
    
    # Find matching visualization files
    old_files = list(old_dir.glob(f"*round_{round_idx}.png"))
    new_files = list(new_dir.glob(f"*round_{round_idx}.png"))
    
    if not old_files or not new_files:
        print(f"Could not find matching visualizations for round {round_idx}")
        print(f"Available rounds in old directory: {sorted([f.name.split('_')[-1].split('.')[0] for f in old_dir.glob('*.png')])}")
        print(f"Available rounds in new directory: {sorted([f.name.split('_')[-1].split('.')[0] for f in new_dir.glob('*.png')])}")
        sys.exit(1)
    
    old_viz_path = old_files[0]
    new_viz_path = new_files[0]
    
    print(f"Comparing visualizations for round {round_idx}")
    print(f"Old: {old_viz_path}")
    print(f"New: {new_viz_path}")
    
    compare_visualizations(old_viz_path, new_viz_path)
