#based on the previous readingDatabase.py script, updated to handle velocity data

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox

#load images and velocities from disk
images = np.load("output/Town07_400s_images.npy")         #shape (N, H, W, 3) or (N, 3, H, W)
velocities = np.load("output/car_velocity.npy")           #shape (N,)

#if images are channels-first (C,H,W), switch to channels-last (H,W,C) for display
if images.ndim == 4 and images.shape[1] == 3:
    images = np.transpose(images, (0, 2, 3, 1))

#ensure uint8 for clean rendering in matplotlib
if images.max() <= 1.0:
    images = (images * 255).astype(np.uint8)
else:
    images = images.astype(np.uint8)

#index we’re currently viewing; list so handlers can modify it
index = [0]
#how many frames to jump when skipping
skip_step = 10

fig, ax = plt.subplots(figsize=(6, 4))
#leave some room at the bottom for buttons and the textbox
plt.subplots_adjust(bottom=0.3)

def show_image(idx):
    #draw the current frame with its velocity
    ax.clear()
    img = images[idx]
    velocity = velocities[idx]
    ax.imshow(img)
    ax.set_title(f"Index: {idx} | Velocity: {velocity:.2f} m/s")
    ax.axis('off')
    plt.draw()

def next_image(event):
    #move forward by one frame
    index[0] = (index[0] + 1) % len(images)
    show_image(index[0])

def prev_image(event):
    #move backward by one frame
    index[0] = (index[0] - 1) % len(images)
    show_image(index[0])

def skip_forward(event):
    #jump ahead by skip_step frames
    index[0] = (index[0] + skip_step) % len(images)
    show_image(index[0])

def skip_back(event):
    #jump back by skip_step frames
    index[0] = (index[0] - skip_step) % len(images)
    show_image(index[0])

def goto_index(text):
    #jump to an index typed in the textbox; ignore invalid input
    try:
        i = int(text)
        index[0] = i % len(images)
        show_image(index[0])
    except ValueError:
        pass

#navigation buttons
axprev  = plt.axes([0.05, 0.15, 0.1, 0.075])
axnext  = plt.axes([0.17, 0.15, 0.1, 0.075])
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

#index textbox for direct jumps
axtbox = plt.axes([0.60, 0.15, 0.2, 0.075])
text_box = TextBox(axtbox, 'Go to idx', initial=str(index[0]))
text_box.on_submit(goto_index)

#render the first frame
show_image(index[0])
plt.show()
