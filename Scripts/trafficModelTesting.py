# traffic_light_viewer.py
# Interactive viewer for traffic light classification results (aligned to training)

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox

# ==== Paths (align with your training script) ====
IMAGES_PATH = "output/traffic_lights_images.npy"
LABELS_PATH = "output/traffic_lights_label.npy"  # note: singular 'label' in training
SAVE_DIR    = "Models"
WEIGHTS_PTH = os.path.join(SAVE_DIR, "traffic_classifier_state.pth")
CLASS_MAP_JSON = os.path.join(SAVE_DIR, "class_mapping.json")

# ==== Label semantics ====
# Default fallback mapping (will be overwritten by class_mapping.json if found)
label_to_index = {-1: 0, 0: 1, 1: 2}
index_to_label = {v: k for k, v in label_to_index.items()}

CLASS_NAMES = {-1: "No Light", 0: "Red", 1: "Green"}
CLASS_EMOJI = {-1: "⚫️",       0: "🔴",  1: "🟢"}

# Try to load class mapping from training artifacts
if os.path.exists(CLASS_MAP_JSON):
    with open(CLASS_MAP_JSON, "r") as f:
        m = json.load(f)
    if "label_to_index" in m and "index_to_label" in m:
        # keys were dumped as strings; convert back
        label_to_index = {int(k): int(v) for k, v in m["label_to_index"].items()}
        index_to_label = {int(k): int(v) for k, v in m["index_to_label"].items()}

# ==== Load numpy data ====
if not (os.path.exists(IMAGES_PATH) and os.path.exists(LABELS_PATH)):
    raise FileNotFoundError("Check IMAGES_PATH and LABELS_PATH")

X = np.load(IMAGES_PATH)  # (N,H,W) or (N,H,W,C)
y_raw = np.load(LABELS_PATH)  # (N,) with values in {-1,0,1}
if X.ndim not in (3, 4):
    raise ValueError(f"Expected images with 3 or 4 dims (N,H,W[,C]); got {X.shape}")

# Ensure channel dimension and channel-first format (same as training)
if X.ndim == 3:
    X = np.expand_dims(X, axis=1)            # (N,1,H,W)
else:
    X = np.transpose(X, (0, 3, 1, 2))        # (N,C,H,W)

# Normalize to [0,1] (same as training)
X = X.astype("float32")
if X.max() > 1.5:
    X /= 255.0

# Map GT labels {-1,0,1} -> {0,1,2} for compatibility checks (we'll still *display* -1/0/1)
y = np.vectorize(label_to_index.__getitem__)(y_raw)

N, C, H, W = X.shape
if N == 0:
    raise RuntimeError("No samples found.")

# For display (convert to HWC uint8)
def chw_float_to_hwc_uint8(x_chw):
    x = x_chw
    if x.shape[0] == 1:
        x = np.repeat(x, 3, axis=0)  # grayscale -> 3ch
    x = np.transpose(x, (1, 2, 0))   # CHW -> HWC
    x = (x * 255.0).clip(0, 255).astype(np.uint8)
    return x

images_display = np.stack([chw_float_to_hwc_uint8(X[i]) for i in range(N)], axis=0)

# ==== Build model exactly like training ====
def build_resnet_classifier(in_channels: int, num_classes: int):
    try:
        from torchvision.models import resnet18
        backbone = resnet18(weights=None)  # new API path
    except Exception:
        from torchvision.models import resnet18
        backbone = resnet18(pretrained=False)

    if in_channels != 3:
        backbone.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
    in_feat = backbone.fc.in_features
    backbone.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_feat, num_classes)
    )

    class ResNetClassifier(nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net
        def forward(self, x):
            # Upscale to 224 if needed (same as training)
            if x.dim() == 4:
                _, _, H, W = x.shape
                if H < 224 or W < 224:
                    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
            return self.net(x)

    return ResNetClassifier(backbone)

num_classes = 3  # fixed by your training
in_channels = C

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_resnet_classifier(in_channels=in_channels, num_classes=num_classes).to(device)
model.eval()

# ==== Load weights (handles both raw state_dict or 'best_state' dict) ====
if not os.path.exists(WEIGHTS_PTH):
    raise FileNotFoundError(f"Weights not found: {WEIGHTS_PTH}")
state = torch.load(WEIGHTS_PTH, map_location=device)
if isinstance(state, dict) and "model_state_dict" in state:
    state_dict = state["model_state_dict"]
else:
    state_dict = state
model.load_state_dict(state_dict, strict=True)
model.eval()

# ==== Single-image prediction (numpy -> model -> label -1/0/1) ====
def predict_one(model, np_chw_image, device):
    """
    np_chw_image: (C,H,W) float32 in [0,1]
    returns: (pred_label_in_{-1,0,1}, conf_float, probs_tensor_over_indices)
    """
    x = np_chw_image
    if x.ndim != 3:
        raise ValueError("Expected CHW image")
    # Channel alignment to training channels
    if x.shape[0] != in_channels:
        if in_channels == 1 and x.shape[0] == 3:
            x = x.mean(axis=0, keepdims=True)  # RGB -> gray
        elif in_channels == 3 and x.shape[0] == 1:
            x = np.repeat(x, 3, axis=0)        # gray -> RGB
        else:
            raise ValueError(f"Channel mismatch: expected {in_channels}, got {x.shape[0]}")
    # To tensor
    xt = torch.from_numpy(x).unsqueeze(0).to(device)  # (1,C,H,W)
    with torch.no_grad():
        _, _, Ht, Wt = xt.shape
        if Ht < 224 or Wt < 224:
            xt = F.interpolate(xt, size=(224, 224), mode="bilinear", align_corners=False)
        logits = model(xt)
        probs = torch.softmax(logits, dim=1).squeeze(0)  # over indices {0,1,2}
        conf, pred_idx = torch.max(probs, dim=0)
        pred_label = index_to_label[int(pred_idx.item())]  # map back to {-1,0,1}
    return int(pred_label), float(conf.item()), probs.cpu()

# ==== UI ====
index = [0]
skip_step = 10

fig, ax = plt.subplots(figsize=(6.8, 4.8))
plt.subplots_adjust(bottom=0.3)

def show_image(idx):
    ax.clear()
    img = images_display[idx]
    gt_label = int(y_raw[idx])  # already in {-1,0,1}
    pred_label, conf, probs = predict_one(model, X[idx], device)

    pred_name = CLASS_NAMES.get(pred_label, str(pred_label))
    gt_name   = CLASS_NAMES.get(gt_label, str(gt_label))
    emoji     = CLASS_EMOJI.get(pred_label, "")

    ax.imshow(img)
    ax.axis('off')
    ax.set_title(
        f"Index: {idx}/{N-1}  |  GT: {gt_name}  |  Pred: {emoji} {pred_name}  "
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

# Buttons
axprev  = plt.axes([0.05, 0.15, 0.1, 0.075])
axnext  = plt.axes([0.17, 0.15, 0.1, 0.075])
axskipb = plt.axes([0.29, 0.15, 0.1, 0.075])
axskipf = plt.axes([0.41, 0.15, 0.1, 0.075])

btn_prev  = plt.Button(axprev,  'Prev')
btn_next  = plt.Button(axnext,  'Next')
btn_skipb = plt.Button(axskipb, f'-{skip_step}')
btn_skipf = plt.Button(axskipf, f'+{skip_step}')

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
