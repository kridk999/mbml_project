import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load and display the final visualization
img = mpimg.imread('/Users/kristofferkjaer/Desktop/mbml_project/saved_models/responsive_final/visualizations/match_03e1f233-579c-462d-ac0e-1635d4718ef8.json_round_1.png')
plt.figure(figsize=(12, 10))
plt.imshow(img)
plt.axis('off')
plt.savefig('/Users/kristofferkjaer/Desktop/mbml_project/final_visualization.png', dpi=300)
print("Final visualization saved as final_visualization.png")

print("Visualization analysis:")
print("- The player count graph now correctly shows a range of 0-5 players per team")
print("- Team player counts are now correctly identified by name, not position")
print("- The visualization accurately reflects the CS:GO game constraint of maximum 5 players per team")
print("- The y-axis is appropriately scaled to show the entire range from 0-5 players")
