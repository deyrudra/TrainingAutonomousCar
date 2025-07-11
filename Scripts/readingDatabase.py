import numpy as np
import matplotlib.pyplot as plt

images = np.load("output/images.npy")
angles = np.load("output/angles.npy")

idx = 310 #the index of the image you want to visualize
plt.imshow(images[idx], cmap='gray')
plt.title(f"Steering Angle: {angles[idx]:.2f}")
plt.axis('off')
plt.show()
