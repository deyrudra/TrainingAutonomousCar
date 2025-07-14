import numpy as np
import matplotlib.pyplot as plt

# Load files
images = np.load("output/images.npy")
angles = np.load("output/angles.npy")
turn_signals = np.load("output/turn_signals.npy")

# Choose index to check
idx = 3045

# Sanity check
if idx >= len(images):
    print(f"Index {idx} is out of bounds (only {len(images)} entries).")
else:
    steering = angles[idx]
    turn = int(turn_signals[idx])  # -1, 0, 1

    # Turn label for readability
    turn_label = "Left" if turn == -1 else "Right" if turn == 1 else "None"

    print(f"Index: {idx}")
    print(f"Steering Angle: {steering:.2f}")
    print(f"Turn Signal: {turn_label} ({turn})")

    # Show image
    plt.imshow(images[idx], cmap='gray')
    plt.title(f"Steering: {steering:.2f} | Turn: {turn_label}")
    plt.axis('off')
    plt.show()