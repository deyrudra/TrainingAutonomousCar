import os
import numpy as np

# Inputs(the two files)
IMG1 = r"C:/carla_data/traffic_lights_run/intersectionlight.npy"
LAB1 = r"C:/carla_data/traffic_lights_run/labels.npy"
IMG2 = r"C:/carla_data/traffic_lights_run_2/intersectionlight2.npy"
LAB2 = r"C:/carla_data/traffic_lights_run_2/labels2.npy"

# Output (creates a third file, doesn't affect the original two)
OUT_IMG = r"C:/carla_data/traffic_lights_run/intersectionlight_merged.npy"
OUT_LAB = r"C:/carla_data/traffic_lights_run/labels_merged.npy"

imgs1 = np.load(IMG1); labs1 = np.load(LAB1)
imgs2 = np.load(IMG2); labs2 = np.load(LAB2)

assert imgs1.shape[0] == labs1.shape[0]
assert imgs2.shape[0] == labs2.shape[0]
assert imgs1.shape[1:] == imgs2.shape[1:] == (240, 320, 3)

# Add the two files together
imgs_all = np.concatenate([imgs1.astype(np.uint8, copy=False),
                           imgs2.astype(np.uint8, copy=False)], axis=0)
labs_all = np.concatenate([labs1.astype(np.int8, copy=False),
                           labs2.astype(np.int8, copy=False)], axis=0)


np.save(OUT_IMG, imgs_all)
np.save(OUT_LAB, labs_all)

print(f"merged images -> {OUT_IMG} (N={imgs_all.shape[0]})")
print(f"merged labels -> {OUT_LAB} (N={labs_all.shape[0]})")
