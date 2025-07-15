import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# Load files
images = np.load("output/std_images.npy")  # shape (N, 3, 224, 224) or (N, H, W, 3)
angles = np.load("output/std_angles.npy")
turn_signals = np.load("output/std_turn_signals.npy")

# Convert shape if needed (C,H,W) → (H,W,C)
if images.ndim == 4 and images.shape[1] == 3:
    images = np.transpose(images, (0, 2, 3, 1))

# Normalize images for display (uint8)
if images.max() <= 1.0:
    images = (images * 255).astype(np.uint8)
else:
    images = images.astype(np.uint8)

index = [0]  # mutable index holder

fig, ax = plt.subplots(figsize=(6, 4))
plt.subplots_adjust(bottom=0.2)  # space for buttons

def show_image(idx):
    ax.clear()
    img = images[idx]
    steering = angles[idx]
    turn = int(turn_signals[idx])
    turn_label = "Left" if turn == -1 else "Right" if turn == 1 else "None"
    ax.imshow(img)
    ax.set_title(f"Index: {idx} | Steering: {steering:.2f} | Turn: {turn_label}")
    ax.axis('off')
    plt.draw()

def next_image(event):
    index[0] = (index[0] + 1) % len(images)
    show_image(index[0])

def prev_image(event):
    index[0] = (index[0] - 1) % len(images)
    show_image(index[0])

# Buttons for navigation
axprev = plt.axes([0.3, 0.05, 0.1, 0.075])
axnext = plt.axes([0.6, 0.05, 0.1, 0.075])

btn_prev = Button(axprev, 'Previous')
btn_next = Button(axnext, 'Next')

btn_prev.on_clicked(prev_image)
btn_next.on_clicked(next_image)

# Show first image
show_image(index[0])
plt.show()
