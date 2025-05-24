import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load and display the first image
img = mpimg.imread('/Users/kristofferkjaer/Desktop/mbml_project/saved_models/responsive_fixed2/visualizations/match_03e1f233-579c-462d-ac0e-1635d4718ef8.json_round_1.png')
plt.figure(figsize=(12, 10))
plt.imshow(img)
plt.axis('off')
plt.savefig('/Users/kristofferkjaer/Desktop/mbml_project/fixed_visualization.png', dpi=300)
print("Fixed visualization saved as fixed_visualization.png")
