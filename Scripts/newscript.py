import os
import json
import time
import random
import numpy as np
import cv2
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional: if torchvision is available, we'll use resnet18
try:
    from torchvision.models import resnet18
    _HAS_TORCHVISION = True
except Exception:
    _HAS_TORCHVISION = False

# CARLA must be installed and a simulator must be running
import carla

"""
This script COMBINES manual CARLA driving + dataset capture with LIVE traffic-light
classification and an on-screen HUD overlay.

It reuses your training viewer's model definition/loading and your driving & capture
loop. While you drive, the top-left HUD will show the predicted light (🔴/🟢/⚫️)
plus confidence.

Controls (keyboard):
  - W / S / A / D : throttle / brake or reverse / steer
  - Q / E         : left / right indicators
  - T             : teleport to a random intersection
  - R             : force all lights RED  + capture N frames labeled 0
  - G             : force all lights GREEN+ capture N frames labeled 1
  - V             : try to force all lights OFF (or normal) + capture N frames labeled -1
  - F             : flip car 180°
  - ESC           : quit

Gamepad (Xbox-like) supported similarly (see printout at startup).
"""

# =====================
# ==== PATHS/IO =======
# =====================
IMAGES_FILE = "output/traffic_lights_images.npy"
LABELS_FILE = "output/traffic_lights_label.npy"  # singular to match your training artifacts
SAVE_DIR    = "Models"
WEIGHTS_PTH = os.path.join(SAVE_DIR, "traffic_classifier_state.pth")
CLASS_MAP_JSON = os.path.join(SAVE_DIR, "class_mapping.json")

# ===============================
# ==== CLASS/LABEL MAPPINGS  ====
# ===============================
# Default fallback mapping; will be overwritten by class_mapping.json if present
label_to_index = {-1: 0, 0: 1, 1: 2}
index_to_label = {v: k for k, v in label_to_index.items()}

CLASS_NAMES = {-1: "No Light", 0: "Red", 1: "Green"}
CLASS_EMOJI = {-1: "⚫️",       0: "🔴",   1: "🟢"}

if os.path.exists(CLASS_MAP_JSON):
    try:
        with open(CLASS_MAP_JSON, "r") as f:
            m = json.load(f)
        if "label_to_index" in m and "index_to_label" in m:
            label_to_index = {int(k): int(v) for k, v in m["label_to_index"].items()}
            index_to_label = {int(k): int(v) for k, v in m["index_to_label"].items()}
    except Exception as e:
        print(f"Warning: failed to read {CLASS_MAP_JSON}: {e}")

# ===========================
# ==== MODEL DEFINITION  ====
# ===========================

def build_resnet_classifier(in_channels: int, num_classes: int):
    if not _HAS_TORCHVISION:
        raise RuntimeError("torchvision is required for resnet18. Please pip install torchvision.")

    # Prefer new API path; fall back to deprecated arg if needed
    try:
        backbone = resnet18(weights=None)
    except Exception:
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

# Create and load the model
num_classes = 3
in_channels = 3  # your CARLA camera is RGB; training artifacts also saved RGB

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_resnet_classifier(in_channels=in_channels, num_classes=num_classes).to(device)
model.eval()

if not os.path.exists(WEIGHTS_PTH):
    raise FileNotFoundError(f"Weights not found: {WEIGHTS_PTH}")
state = torch.load(WEIGHTS_PTH, map_location=device)
state_dict = state.get("model_state_dict", state)
model.load_state_dict(state_dict, strict=True)
model.eval()

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

@torch.inference_mode()
def predict_one(np_chw_image: np.ndarray):
    x = np_chw_image.astype(np.float32)
    if x.max() > 1.5:  # guard if uint8 slipped in
        x = x / 255.0
    if x.shape[0] == 1:
        x = np.repeat(x, 3, axis=0)
    # resize to 224x224 if needed (your model will upsample, but do it here for correctness)
    C, H, W = x.shape
    if H != 224 or W != 224:
        x_hw = np.transpose(x, (1,2,0))  # HWC
        x_hw = cv2.resize(x_hw, (224,224), interpolation=cv2.INTER_LINEAR)
        x = np.transpose(x_hw, (2,0,1))
    # normalize
    x[0] = (x[0] - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
    x[1] = (x[1] - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
    x[2] = (x[2] - IMAGENET_MEAN[2]) / IMAGENET_STD[2]

    xt = torch.from_numpy(x).unsqueeze(0).to(device)
    logits = model(xt)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    pred_idx = int(np.argmax(probs))
    conf = float(probs[pred_idx])
    pred_label = int(index_to_label[pred_idx])
    return pred_label, conf, probs

# @torch.inference_mode()
# def predict_one(np_chw_image: np.ndarray):
#     """np_chw_image: (C,H,W) float32 in [0,1] -> (pred_label -1/0/1, conf_float, probs[3])"""
#     x = np_chw_image
#     if x.ndim != 3:
#         raise ValueError("Expected CHW image")
#     if x.shape[0] != in_channels:
#         # Simple channel alignment just in case
#         if in_channels == 1 and x.shape[0] == 3:
#             x = x.mean(axis=0, keepdims=True)
#         elif in_channels == 3 and x.shape[0] == 1:
#             x = np.repeat(x, 3, axis=0)
#         else:
#             raise ValueError(f"Channel mismatch: expected {in_channels}, got {x.shape[0]}")

#     xt = torch.from_numpy(x).unsqueeze(0).to(device)
#     logits = model(xt)
#     probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
#     pred_idx = int(np.argmax(probs))
#     conf = float(probs[pred_idx])
#     pred_label = int(index_to_label[pred_idx])
#     return pred_label, conf, probs

# =============================
# ==== IMAGE PREPROCESSING ====
# =============================

def traffic_light_crop(img: np.ndarray) -> np.ndarray:
    """
    Same center-zoom + bottom-third blackout you use for data capture.
    Input: HxWx3 RGB uint8; Output: HxWx3 RGB uint8
    """
    if img is None or img.ndim != 3:
        return img

    h, w = img.shape[:2]
    out = img

    # Center zoom (~50%)
    z = 1.0 + (50.0 / 100.0)
    new_w = max(1, int(w / z))
    new_h = max(1, int(h / z))
    x1 = max(0, (w - new_w) // 2)
    y1 = max(0, (h - new_h) // 2)
    cropped = out[y1:y1 + new_h, x1:x1 + new_w]
    if cropped.size > 0:
        out = cv2.resize(cropped, (w, h))

    # Blackout bottom third to hide hood/road
    out = out.copy()
    out[(2*h)//3:, :, :] = 0
    return out


def rgb_to_chw_float01(img_rgb: np.ndarray) -> np.ndarray:
    """HxWx3 uint8 -> 3xHxW float32 in [0,1]"""
    x = img_rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    return x

# ======================
# ==== PYGAME UI  ======
# ======================
pygame.init()
width, height = 1024, 768
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("CARLA Manual Control + Live TL Prediction")

# Basic font for HUD
try:
    font = pygame.font.SysFont("consolas", 24)
except Exception:
    font = pygame.font.Font(None, 24)

# ======================
# ==== CARLA SETUP  ====
# ======================
client = carla.Client("localhost", 2000)
client.set_timeout(15.0)
client.load_world("Town03")
world = client.get_world()
blueprint_library = world.get_blueprint_library()
map_ = world.get_map()

# === Spawn a vehicle ===
def _spawn_vehicle(world, max_tries=25):
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.filter("vehicle.*model3*")[0]
    spawns = world.get_map().get_spawn_points()
    random.shuffle(spawns)
    vehicle = None
    tries = 0
    for sp in spawns:
        vehicle = world.try_spawn_actor(vehicle_bp, sp)
        if vehicle:
            break
        tries += 1
        if tries >= max_tries:
            break
    if not vehicle:
        sp = random.choice(spawns)
        vehicle = world.spawn_actor(vehicle_bp, sp)
    vehicle.set_autopilot(False)
    return vehicle

vehicle = _spawn_vehicle(world)

# === Spectator camera ===
spectator = world.get_spectator()

# === Attach camera sensor ===
def _attach_camera(world, vehicle):
    bp_lib = world.get_blueprint_library()
    camera_bp = bp_lib.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '1024')
    camera_bp.set_attribute('image_size_y', '768')
    camera_bp.set_attribute('fov', '110')
    camera_bp.set_attribute('sensor_tick', '0.1')  # 10 FPS
    cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    cam = world.spawn_actor(camera_bp, cam_transform, attach_to=vehicle)
    return cam

camera = _attach_camera(world, vehicle)

# === Camera feed state ===
camera_image_surface = None
camera_frame_rgb_native = None  # HxWx3 RGB uint8

# === Dataset capture state ===
CAPTURE_FRAMES = 7
capture_active = False
capture_remaining = 0
capture_label_value = None
capture_buffer_images = []
capture_buffer_labels = []


def _ensure_output_dir():
    out_dir = os.path.dirname(IMAGES_FILE) or "."
    os.makedirs(out_dir, exist_ok=True)


def save_dataset(images_list, labels_list):
    _ensure_output_dir()
    new_imgs = np.stack(images_list, axis=0).astype(np.uint8)  # (N, 224, 224, 3)
    new_lbls = np.asarray(labels_list, dtype=np.int64)

    if os.path.exists(IMAGES_FILE) and os.path.exists(LABELS_FILE):
        try:
            old_imgs = np.load(IMAGES_FILE)
            old_lbls = np.load(LABELS_FILE)
            imgs = np.concatenate([old_imgs, new_imgs], axis=0)
            lbls = np.concatenate([old_lbls, new_lbls], axis=0)
        except Exception:
            imgs, lbls = new_imgs, new_lbls
    else:
        imgs, lbls = new_imgs, new_lbls

    np.save(IMAGES_FILE, imgs)
    np.save(LABELS_FILE, lbls)

    counts = {label: int(np.sum(lbls == label)) for label in [-1, 0, 1]}

    print(f"Saved {len(new_lbls)} frames.")
    print(f"Totals -> images: {imgs.shape[0]}, labels: {lbls.shape[0]}")
    print(f"Label counts: NO LIGHT (-1): {counts[-1]}, RED (0): {counts[0]}, GREEN (1): {counts[1]}")


def start_capture(label_value):
    global capture_active, capture_remaining, capture_label_value
    if capture_active:
        print("Capture already in progress—wait for it to finish.")
        return
    capture_active = True
    capture_remaining = CAPTURE_FRAMES
    capture_label_value = label_value
    capture_buffer_images.clear()
    capture_buffer_labels.clear()
    print(f"Capture started for label {label_value} (next {CAPTURE_FRAMES} frames).")


# ======================
# ==== SENSOR CALLBACK =
# ======================

def process_image(image):
    global camera_image_surface, camera_frame_rgb_native, capture_active, capture_remaining
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))[:, :, :3][:, :, ::-1]  # BGRA->BGR->RGB

    array = traffic_light_crop(array)

    camera_frame_rgb_native = array  # Keep most recent cropped frame
    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    camera_image_surface = pygame.transform.scale(surface, (width, height))

    if capture_active and capture_remaining > 0:
        resized_img = cv2.resize(camera_frame_rgb_native, (224, 224))
        capture_buffer_images.append(resized_img.copy())
        capture_buffer_labels.append(capture_label_value)
        capture_remaining -= 1
        print(f"Captured frame {CAPTURE_FRAMES - capture_remaining}/{CAPTURE_FRAMES} (label={capture_label_value})")
        if capture_remaining == 0:
            save_dataset(capture_buffer_images, capture_buffer_labels)
            print("Capture finished.")
            capture_active = False

camera.listen(process_image)

# =========================
# ==== TRAFFIC LIGHT API ==
# =========================
TLS = carla.TrafficLightState
_HAS_FREEZE = hasattr(carla.TrafficLight, "set_time_is_frozen")
_HAS_OFF = hasattr(TLS, "Off")


def force_all_lights(state: carla.TrafficLightState):
    lights = world.get_actors().filter('traffic.traffic_light')
    count = 0
    for tl in lights:
        try:
            tl.set_state(state)
            if _HAS_FREEZE:
                tl.set_time_is_frozen(True)
            else:
                if state == TLS.Red:
                    tl.set_red_time(1e6); tl.set_yellow_time(0.001); tl.set_green_time(0.001)
                elif state == TLS.Green:
                    tl.set_green_time(1e6); tl.set_yellow_time(0.001); tl.set_red_time(0.001)
                else:
                    tl.set_yellow_time(1e6); tl.set_green_time(0.001); tl.set_red_time(0.001)
        except Exception:
            pass
        count += 1
    print(f"Forced {count} traffic lights to {state.name}.")


def force_all_lights_off():
    lights = world.get_actors().filter('traffic.traffic_light')
    turned_off = 0
    fallback = 0
    for tl in lights:
        try:
            if _HAS_OFF:
                tl.set_state(TLS.Off)
                if _HAS_FREEZE:
                    tl.set_time_is_frozen(True)
                turned_off += 1
            else:
                if _HAS_FREEZE:
                    tl.set_time_is_frozen(False)
                tl.set_green_time(10.0); tl.set_yellow_time(3.0); tl.set_red_time(10.0)
                fallback += 1
        except Exception:
            pass
    if turned_off:
        print(f"Set {turned_off} traffic lights to OFF.")
    if fallback:
        print("OFF not supported on this CARLA; reverted some lights to normal timing.")


def unforce_all_lights():
    try:
        lights = world.get_actors().filter('traffic.traffic_light')
        for tl in lights:
            try:
                if _HAS_FREEZE:
                    tl.set_time_is_frozen(False)
                tl.set_green_time(10.0); tl.set_yellow_time(3.0); tl.set_red_time(10.0)
            except Exception:
                pass
    except Exception:
        pass

# === Teleport & flip ===

def teleport_to_intersection():
    waypoints = [wp for wp in map_.generate_waypoints(2.0) if wp.is_junction]
    if not waypoints:
        print("No intersections found!")
        return
    wp = random.choice(waypoints)
    tf = wp.transform
    tf.location.z += 0.5
    vehicle.set_transform(tf)
    vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
    vehicle.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
    print(f"Teleported to intersection at {tf.location}")


def flip_car_180():
    tf = vehicle.get_transform()
    tf.rotation.yaw = (tf.rotation.yaw + 180.0) % 360.0
    vehicle.set_transform(tf)
    print("Vehicle flipped 180 degrees.")

# ===============================
# ============ GAMEPAD ==========
# ===============================
pygame.joystick.init()
_joystick = None
if pygame.joystick.get_count() > 0:
    _joystick = pygame.joystick.Joystick(0)
    _joystick.init()
    print(f"Gamepad connected: {_joystick.get_name()} | axes={_joystick.get_numaxes()} buttons={_joystick.get_numbuttons()} hats={_joystick.get_numhats()}")
else:
    print("No gamepad detected. Keyboard controls still available.")

_prev_buttons = {}
_prev_hat = (0, 0)

AXIS_LX = 0
AXIS_LY = 1
AXIS_TRIGGERS_COMBINED = 2
AXIS_RY = 3
AXIS_LT_MAYBE = 4
AXIS_RT_MAYBE = 5

BTN_A, BTN_B, BTN_X, BTN_Y = 0, 1, 2, 3
BTN_LB, BTN_RB = 4, 5
BTN_BACK, BTN_START = 6, 7
BTN_LS, BTN_RS = 8, 9


def _deadzone(value: float, dz: float = 0.08) -> float:
    return 0.0 if abs(value) < dz else value


def _axis_or_zero(js, idx):
    try:
        return js.get_axis(idx)
    except Exception:
        return 0.0


def read_gamepad_controls():
    global _prev_buttons, _prev_hat

    if _joystick is None:
        return dict(steer=0.0, throttle=0.0, brake=0.0,
                    left_blinker=False, right_blinker=False, events=[])

    steer = _deadzone(_axis_or_zero(_joystick, AXIS_LX))

    lt = rt = 0.0
    axes = _joystick.get_numaxes()
    if axes >= 6:
        lt = max(0.0, (_axis_or_zero(_joystick, AXIS_LT_MAYBE) + 1.0) * 0.5)
        rt = max(0.0, (_axis_or_zero(_joystick, AXIS_RT_MAYBE) + 1.0) * 0.5)
    else:
        comb = _axis_or_zero(_joystick, AXIS_TRIGGERS_COMBINED)
        rt = max(0.0, comb)
        lt = max(0.0, -comb)

    lt = 0.0 if lt < 0.02 else lt
    rt = 0.0 if rt < 0.02 else rt

    left_blinker = False
    right_blinker = False
    try:
        left_blinker  = _joystick.get_button(BTN_LB) == 1
        right_blinker = _joystick.get_button(BTN_RB) == 1
    except Exception:
        pass

    events = []
    def pressed(btn_idx):
        prev = _prev_buttons.get(btn_idx, 0)
        now = _joystick.get_button(btn_idx)
        _prev_buttons[btn_idx] = now
        return prev == 0 and now == 1

    if pressed(BTN_A):
        events.append("GREEN_CAPTURE")
    if pressed(BTN_B):
        events.append("RED_CAPTURE")
    if pressed(BTN_X):
        events.append("NO_LIGHT_CAPTURE")
    if pressed(BTN_Y):
        events.append("FLIP_180")
    if pressed(BTN_BACK):
        events.append("TELEPORT")

    if _joystick.get_numhats() > 0:
        _prev_hat = _joystick.get_hat(0)

    return dict(
        steer=steer,
        throttle=rt,
        brake=lt,
        left_blinker=left_blinker,
        right_blinker=right_blinker,
        events=events
    )

# ==========================
# ===== HUD / OVERLAY  =====
# ==========================
_last_pred = {
    "label": None,   # -1/0/1
    "conf": 0.0,
    "probs": None,
    "t": 0.0,
}

PRED_EVERY_N_FRAMES = 1  # set >1 if you want to skip frames for speed
_frame_counter = 0


def draw_hud(surface, pred_label, conf, fps):
    emoji = CLASS_EMOJI.get(pred_label, "?")
    name  = CLASS_NAMES.get(pred_label, str(pred_label))
    text  = f"Pred: {emoji} {name}  ({conf*100:.1f}%)   FPS:{fps:.1f}"
    color = (240, 240, 240)

    box = pygame.Surface((580, 36), pygame.SRCALPHA)
    box.fill((0, 0, 0, 140))  # translucent background
    surface.blit(box, (10, 10))
    txt_surf = font.render(text, True, color)
    surface.blit(txt_surf, (20, 16))


# ==========================
# ========= MAIN ===========
# ==========================
clock = pygame.time.Clock()

print("Controls:")
print("Keyboard — W: throttle, S: brake/reverse, A/D: steer, Q/E: turn signals")
print("Keyboard — T: teleport, V: OFF + capture, R: RED + capture, G: GREEN + capture, F: flip 180°, ESC: quit")
print("Gamepad (Xbox One) — Left Stick X: steer, RT: throttle, LT: brake, LB/RB: turn signals")
print("Gamepad — A: GREEN + capture, B: RED + capture, X: OFF/NO LIGHT + capture, Y: flip 180°, Back: teleport")

try:
    while True:
        dt_ms = clock.tick(30)
        fps = 1000.0 / max(1.0, dt_ms)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    raise KeyboardInterrupt
                elif event.key == pygame.K_t:
                    teleport_to_intersection()
                elif event.key == pygame.K_r:
                    force_all_lights(TLS.Red)
                    start_capture(label_value=0)
                elif event.key == pygame.K_g:
                    force_all_lights(TLS.Green)
                    start_capture(label_value=1)
                elif event.key == pygame.K_v:
                    force_all_lights_off()
                    start_capture(label_value=-1)
                elif event.key == pygame.K_f:
                    flip_car_180()

        keys = pygame.key.get_pressed()
        gp = read_gamepad_controls()

        control = carla.VehicleControl()

        vel = vehicle.get_velocity()
        speed = (vel.x**2 + vel.y**2 + vel.z**2) ** 0.5

        kb_throttle = 0.6 if keys[pygame.K_w] else 0.0
        if keys[pygame.K_s]:
            if speed > 0.5:
                kb_brake = 1.0
                kb_reverse = False
                kb_rev_throttle = 0.0
            else:
                kb_brake = 0.0
                kb_reverse = True
                kb_rev_throttle = 0.8
        else:
            kb_brake = 0.0
            kb_reverse = False
            kb_rev_throttle = 0.0

        gp_throttle = float(gp["throttle"])  # 0..1
        gp_brake = float(gp["brake"])        # 0..1

        throttle = max(kb_throttle, gp_throttle)
        brake = max(kb_brake, gp_brake)

        steer = 0.0
        if keys[pygame.K_a]:
            steer -= 0.5
        if keys[pygame.K_d]:
            steer += 0.5
        steer_gp = gp["steer"] * 0.7
        if not (keys[pygame.K_a] or keys[pygame.K_d]):
            steer = steer_gp

        control.left_indicator_light = keys[pygame.K_q] or gp["left_blinker"]
        control.right_indicator_light = keys[pygame.K_e] or gp["right_blinker"]

        control.throttle = throttle
        control.brake = brake
        control.steer = float(np.clip(steer, -1.0, 1.0))
        control.reverse = kb_reverse
        if kb_reverse:
            control.throttle = kb_rev_throttle

        vehicle.apply_control(control)

        # Spectator follow-cam
        transform = vehicle.get_transform()
        forward_vector = transform.get_forward_vector()
        cam_location = transform.location - forward_vector * 8 + carla.Location(z=3)
        cam_rotation = carla.Rotation(pitch=-10, yaw=transform.rotation.yaw)
        spectator.set_transform(carla.Transform(cam_location, cam_rotation))

        # Draw camera feed
        if camera_image_surface is not None:
            screen.blit(camera_image_surface, (0, 0))

        # === LIVE PREDICTION ===
        _frame_counter += 1
        if camera_frame_rgb_native is not None and (_frame_counter % PRED_EVERY_N_FRAMES == 0):
            # Build CHW float [0,1]; model will upsample to 224 if needed
            x_chw = rgb_to_chw_float01(camera_frame_rgb_native)
            pred_label, conf, probs = predict_one(x_chw)
            _last_pred.update({"label": pred_label, "conf": conf, "probs": probs, "t": time.time()})

        if _last_pred["label"] is not None:
            draw_hud(screen, _last_pred["label"], _last_pred["conf"], fps)

        pygame.display.flip()

        # Handle edge-triggered gamepad events AFTER apply_control
        for evt in gp["events"]:
            if evt == "GREEN_CAPTURE":
                force_all_lights(TLS.Green)
                start_capture(label_value=1)
            elif evt == "RED_CAPTURE":
                force_all_lights(TLS.Red)
                start_capture(label_value=0)
            elif evt == "NO_LIGHT_CAPTURE":
                force_all_lights_off()
                start_capture(label_value=-1)
            elif evt == "TELEPORT":
                teleport_to_intersection()
            elif evt == "FLIP_180":
                flip_car_180()

except KeyboardInterrupt:
    print("Exiting and cleaning up...")

finally:
    unforce_all_lights()
    try:
        camera.stop()
    except Exception:
        pass
    try:
        if camera and camera.is_alive:
            camera.destroy()
    except Exception:
        pass
    try:
        if vehicle and vehicle.is_alive:
            vehicle.destroy()
    except Exception:
        pass
    pygame.quit()
    print("Vehicle and camera destroyed. Traffic lights restored. Goodbye!")
