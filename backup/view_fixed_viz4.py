#!/usr/bin/env python3
import sys
import matplotlib.pyplot as plt
from pathlib import Path

def show_visualization(image_path):
    """Display a visualization image with matplotlib."""
    plt.figure(figsize=(12, 10))
    img = plt.imread(image_path)
    plt.imshow(img)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Use the path provided by the user, or use a default one
    if len(sys.argv) > 1:
        viz_path = Path(sys.argv[1])
    else:
        # Default to a specific visualization
        viz_path = Path("./visualizations_raw/03e1f233-579c-462d-ac0e-1635d4718ef8.json_round_1_raw.png")
    
    if not viz_path.exists():
        print(f"Error: Visualization file not found: {viz_path}")
        sys.exit(1)
    
    print(f"Displaying visualization: {viz_path}")
    show_visualization(viz_path)
