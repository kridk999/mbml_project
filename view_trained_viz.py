import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys

# First argument is the image path
if len(sys.argv) < 2:
    print("Please provide an image path as an argument")
    sys.exit(1)

image_path = sys.argv[1]
print(f"Displaying image: {image_path}")
img = mpimg.imread(image_path)
plt.figure(figsize=(12, 10))
plt.imshow(img)
plt.title(f"Visualization: {image_path}")
plt.tight_layout()
plt.show()
