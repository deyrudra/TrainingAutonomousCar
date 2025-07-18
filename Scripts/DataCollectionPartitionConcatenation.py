import numpy as np
import glob
import random
import os

# 1. Find all your “partition” files by prefix
#    e.g. output/Town07_80s_images.npy → prefix=output/Town07_80s
image_files = sorted(glob.glob('../output/partitions/*_images.npy'))
partitions = []
for img_path in image_files:
    prefix = img_path[:-len('_images.npy')]
    ang_path = prefix + '_angles.npy'
    sig_path = prefix + '_turn_signals.npy'
    if os.path.exists(ang_path) and os.path.exists(sig_path):
        partitions.append((img_path, ang_path, sig_path))
    else:
        raise FileNotFoundError(f"Missing angles/signals for {prefix}")

# 2. Shuffle the list of partitions (not the data inside)
random.shuffle(partitions)

# 3. Load & concatenate in that shuffled order
imgs_list, angs_list, sigs_list = [], [], []
for img_path, ang_path, sig_path in partitions:
    imgs_list .append(np.load(img_path))
    angs_list .append(np.load(ang_path))
    sigs_list .append(np.load(sig_path))

all_images  = np.concatenate(imgs_list, axis=0)
all_angles  = np.concatenate(angs_list, axis=0)
all_signals = np.concatenate(sigs_list, axis=0)

print("Final shapes:", all_images.shape, all_angles.shape, all_signals.shape)

# 4. (Optional) Save out your big arrays
np.save('../output/all_images.npy',  all_images)
np.save('../output/all_angles.npy',  all_angles)
np.save('../output/all_signals.npy', all_signals)