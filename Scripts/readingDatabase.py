import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox

# load all data files
images = np.load("output/all_images.npy")
angles = np.load("output/all_angles.npy")
turn_signals = np.load("output/all_turn_signals.npy")

# Convert shape if needed (C,H,W) → (H,W,C)
if images.ndim == 4 and images.shape[1] == 3:
    images = np.transpose(images, (0, 2, 3, 1))

# normalize the images
if images.max() <= 1.0:
    images = (images * 255).astype(np.uint8)
else:
    images = images.astype(np.uint8)

# settings for index navigation
index = [0]
skip_step = 10

fig, ax = plt.subplots(figsize=(6, 4))
plt.subplots_adjust(bottom=0.3)

# displaying image along with corresponding data
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

def skip_forward(event):
    index[0] = (index[0] + skip_step) % len(images)
    show_image(index[0])

def skip_back(event):
    index[0] = (index[0] - skip_step) % len(images)
    show_image(index[0])

def goto_index(text):
    try:
        i = int(text)
        index[0] = i % len(images)
        show_image(index[0])
    except ValueError:
        pass

# Navigation buttons for the GUI
axprev = plt.axes([0.05, 0.15, 0.1, 0.075])
axnext = plt.axes([0.17, 0.15, 0.1, 0.075])
axskipb = plt.axes([0.29, 0.15, 0.1, 0.075])
axskipf = plt.axes([0.41, 0.15, 0.1, 0.075])

btn_prev  = Button(axprev,  'Prev')
btn_next  = Button(axnext,  'Next')
btn_skipb = Button(axskipb, f'-{skip_step}')
btn_skipf = Button(axskipf, f'+{skip_step}')

btn_prev.on_clicked(prev_image)
btn_next.on_clicked(next_image)
btn_skipb.on_clicked(skip_back)
btn_skipf.on_clicked(skip_forward)

# Index skipper TextBox
axtbox = plt.axes([0.60, 0.15, 0.2, 0.075])
text_box = TextBox(axtbox, 'Go to idx', initial=str(index[0]))
text_box.on_submit(goto_index)

# Show first image
show_image(index[0])
plt.show()
