import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Load the original and fixed visualizations
try:
    # Try to load an original visualization (if available)
    img_original = mpimg.imread('/Users/kristofferkjaer/Desktop/mbml_project/saved_models/responsive/visualizations/match_03e1f233-579c-462d-ac0e-1635d4718ef8.json_round_1.png')
    has_original = True
except (FileNotFoundError, IOError):
    has_original = False
    print("Original visualization not found, will only show fixed version")

# Load the fixed visualization
img_fixed = mpimg.imread('/Users/kristofferkjaer/Desktop/mbml_project/saved_models/responsive_final/visualizations/match_03e1f233-579c-462d-ac0e-1635d4718ef8.json_round_1.png')

# Create comparison figure
if has_original:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    ax1.imshow(img_original)
    ax1.set_title("Original Visualization\n(Incorrect player counts - not limited to 0-5)", fontsize=14)
    ax1.axis('off')
    
    ax2.imshow(img_fixed)
    ax2.set_title("Fixed Visualization\n(Correct player counts - properly scaled to 0-5)", fontsize=14)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('/Users/kristofferkjaer/Desktop/mbml_project/visualization_comparison.png', dpi=300)
    print("Comparison visualization saved as visualization_comparison.png")
else:
    # Just show the fixed version
    plt.figure(figsize=(12, 10))
    plt.imshow(img_fixed)
    plt.title("Fixed Visualization\n(Correct player counts - properly scaled to 0-5)", fontsize=14)
    plt.axis('off')
    plt.savefig('/Users/kristofferkjaer/Desktop/mbml_project/fixed_visualization.png', dpi=300)
    print("Fixed visualization saved as fixed_visualization.png")

print("\nVisualization Fix Summary:")
print("---------------------------")
print("1. Problem: The original visualization code used hardcoded indices for 'CT players alive' (index 0) ")
print("   and 'T players alive' (index 1), which led to incorrect player counts when columns were reordered.")
print("2. Solution: The fix identifies player count columns by name instead of position:")
print("   - Searches for 'ct' and 'alive' in column names for CT players")
print("   - Searches for 't_alive' for T players (avoiding 'ct' in the name)")
print("3. Additional Improvement: Added y-axis limits (0-5) to match CS:GO's maximum team size")
print("4. Result: The visualizations now correctly show player counts with appropriate scaling,")
print("   making the data display accurate regardless of column ordering in the feature extraction process.")
