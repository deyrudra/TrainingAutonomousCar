# Interactive viewer for traffic light classification results.
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from torchvision import models, transforms
from PIL import Image

IMAGES_PATH = "output/intersectionlight_images_merged.npy"
LABELS_PATH = "output/intersectionlight_labels_merged.npy"
WEIGHTS_PATH = "Models/traffic_lights_model.pt"
RESIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

CLASS_NAMES = {0: "Red", 1: "Green"}
CLASS_EMOJI  = {0: "🔴", 1: "🟢"}

# ==== Model (matches training architecture) ====
def build_model(num_classes=2):
   
    try:
        
        model = models.resnet18(weights=None)
    except TypeError:
        model = models.resnet18(pretrained=False)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    return model

# ==== Transforms (same stats as training) ====
eval_transform = transforms.Compose([
    transforms.Resize((RESIZE, RESIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

def to_pil_uint8(arr):
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr)

    #if CHW -> HWC
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))

    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)

    # If single channel, convert to 3 by repeat
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)

    # If grayscale 2D, stack to 3 channels
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)

    return Image.fromarray(arr)

def predict_one(model, image_array, device):
    model.eval()
    with torch.no_grad():
        pil = to_pil_uint8(image_array)
        x = eval_transform(pil).unsqueeze(0).to(device)
        logits = model(x)
        probs = F.softmax(logits, dim=1).squeeze(0)
        conf, pred = torch.max(probs, dim=0)
    return int(pred.item()), float(conf.item()), probs.cpu()

# ==== Load model + data ====
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = build_model(num_classes=2).to(device)

if not os.path.exists(WEIGHTS_PATH):
    raise FileNotFoundError(f"Weights not found: {WEIGHTS_PATH}")
state = torch.load(WEIGHTS_PATH, map_location=device)
# Support either {'model_state_dict': ...} or raw state_dict
state_dict = state.get('model_state_dict', state)
model.load_state_dict(state_dict)
model.eval()

if not os.path.exists(IMAGES_PATH) or not os.path.exists(LABELS_PATH):
    raise FileNotFoundError("Check IMAGES_PATH and LABELS_PATH")

images = np.load(IMAGES_PATH, mmap_mode='r')
labels = np.load(LABELS_PATH, mmap_mode='r')

# Normalize image array layout/content for display (predict_one handles transforms)
if images.ndim == 4 and images.shape[1] in (1, 3):
    images_display = np.transpose(images[:], (0, 2, 3, 1))
else:
    images_display = images[:]

# If in 0..1 floats, scale for display
if images_display.dtype != np.uint8:
    if images_display.max() <= 1.0:
        images_display = (images_display * 255.0).clip(0, 255).astype(np.uint8)
    else:
        images_display = images_display.clip(0, 255).astype(np.uint8)

N = len(labels)
if N == 0:
    raise RuntimeError("No samples found.")


index = [0]
skip_step = 10

fig, ax = plt.subplots(figsize=(6.4, 4.8))
plt.subplots_adjust(bottom=0.3)

def show_image(idx):
    ax.clear()
    img = images_display[idx]
    gt = int(labels[idx])

    pred_idx, conf, probs = predict_one(model, images[idx], device)
    pred_name = CLASS_NAMES[pred_idx]
    gt_name = CLASS_NAMES.get(gt, str(gt))
    emoji = CLASS_EMOJI.get(pred_idx, "")

    ax.imshow(img)
    ax.axis('off')
    ax.set_title(
        f"Index: {idx}/{N-1} | GT: {gt_name} | Pred: {emoji} {pred_name} "
        f"(conf {conf*100:.1f}%)"
    )
    plt.draw()

def next_image(event):
    index[0] = (index[0] + 1) % N
    show_image(index[0])

def prev_image(event):
    index[0] = (index[0] - 1) % N
    show_image(index[0])

def skip_forward(event):
    index[0] = (index[0] + skip_step) % N
    show_image(index[0])

def skip_back(event):
    index[0] = (index[0] - skip_step) % N
    show_image(index[0])

def goto_index(text):
    try:
        i = int(text)
        index[0] = i % N
        show_image(index[0])
    except ValueError:
        pass

# ==== Buttons ====
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

axtbox = plt.axes([0.60, 0.15, 0.2, 0.075])
text_box = TextBox(axtbox, 'Go to idx', initial=str(index[0]))
text_box.on_submit(goto_index)

# Initial draw
show_image(index[0])
plt.show()
