import numpy as np

# Input file paths
file1 = "output/traffic_lights_images.npy"
file2 = "output/traffic_lights_label.npy"

# Output file paths for last 20%
file1_last = "output/traffic_lights_images_last20.npy"
file2_last = "output/traffic_lights_label_last20.npy"

# Load arrays
arr1 = np.load(file1)
arr2 = np.load(file2)

# Ensure both have same length along first axis
if arr1.shape[0] != arr2.shape[0]:
    raise ValueError("Arrays must have the same number of rows/elements along axis 0")

# Compute split index
n = arr1.shape[0]
split_index = int(n * 0.8)  # first 80% index

# Split arrays
arr1_last = arr1[split_index:]
arr2_last = arr2[split_index:]

# Save last 20% arrays
np.save(file1_last, arr1_last)
np.save(file2_last, arr2_last)

print(f"Saved last 20% of data to {file1_last} and {file2_last}")
