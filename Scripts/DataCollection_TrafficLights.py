import carla
import pygame
import time
import random
import numpy as np
import os
import cv2  # for resizing to 224x224

# === Pygame setup ===
pygame.init()
width, height = 1024, 768
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("CARLA Manual Control + TL Capture")

# === Connect to CARLA ===
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)
client.load_world("Town05")
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

# === Spectator camera setup ===
spectator = world.get_spectator()

# === Attach Camera Sensor (your settings) ===
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
camera_frame_rgb_native = None

# === Capture state ===
CAPTURE_FRAMES = 7
capture_active = False
capture_remaining = 0
capture_label_value = None
capture_buffer_images = []
capture_buffer_labels = []

# === Output files ===
IMAGES_FILE = "output/traffic_lights_images.npy"
LABELS_FILE = "output/traffic_lights_label.npy"

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

    # Counts for each label
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

def traffic_light_crop(img: np.ndarray) -> np.ndarray:
    """
    Applies a center zoom (by zoom_percent) and optionally blacks out the bottom third.
    Returns a new image with the same HxW as input.
    """
    if img is None or img.ndim != 3:
        return img

    h, w = img.shape[:2]
    out = img

    # ---- Center zoom ----
    z = 1.0 + (50.0 / 100.0)
    new_w = max(1, int(w / z))
    new_h = max(1, int(h / z))
    x1 = max(0, (w - new_w) // 2)
    y1 = max(0, (h - new_h) // 2)
    cropped = out[y1:y1 + new_h, x1:x1 + new_w]
    if cropped.size > 0:
        out = cv2.resize(cropped, (w, h))

    # --- Blackout bottom third ---
    out[h * 2 // 3 :, ...] = 0

    # --- Prepare for color manipulation ---
    is_gray = (out.ndim == 2) or (out.shape[2] == 1)
    if is_gray:
        out_rgb = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    else:
        out_rgb = out

    quarter_w = max(1, w // 4)

    def blur_and_desaturate(region: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(region, (21, 21), 0)
        # hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        # hsv[:, :, 1] = 0
        # desat = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return blurred

    out_rgb[:, :quarter_w, :] = blur_and_desaturate(out_rgb[:, :quarter_w, :])
    out_rgb[:, -quarter_w:, :] = blur_and_desaturate(out_rgb[:, -quarter_w:, :])


    return out  # keep color (we want color for UI), not out_final

def process_image(image):
    global camera_image_surface, camera_frame_rgb_native, capture_active, capture_remaining
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))[:, :, :3][:, :, ::-1]  # BGRA->BGR->RGB

    array = traffic_light_crop(array)

    camera_frame_rgb_native = array
    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    camera_image_surface = pygame.transform.scale(surface, (width, height))

    if capture_active and capture_remaining > 0:
        # Resize to 224x224 BEFORE saving
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

# === Traffic light helpers ===
TLS = carla.TrafficLightState
_HAS_FREEZE = hasattr(carla.TrafficLight, "set_time_is_frozen")
_HAS_OFF = hasattr(TLS, "Off")  # some CARLA versions provide an "Off" state

def force_all_lights(state: carla.TrafficLightState):
    lights = world.get_actors().filter('traffic.traffic_light')
    count = 0
    for tl in lights:
        try:
            tl.set_state(state)
            if _HAS_FREEZE:
                tl.set_time_is_frozen(True)
            else:
                # If we can't freeze the state, bias timings to effectively hold it
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
    """Try to turn lights fully OFF; if not supported, just unforce & set normal cycle."""
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
                # Fallback: unfreeze and reset timings to default-ish behavior
                if _HAS_FREEZE:
                    tl.set_time_is_frozen(False)
                tl.set_green_time(10.0); tl.set_yellow_time(3.0); tl.set_red_time(10.0)
                fallback += 1
        except Exception:
            pass
    if turned_off:
        print(f"Set {turned_off} traffic lights to OFF.")
    if fallback:
        print(f"OFF state not supported on this CARLA build; reverted {fallback} lights to normal timing.")

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

# === NEW: Helpers to set perpendicular RED, parallel GREEN on L1/LB ===
def _normalize180(deg: float) -> float:
    while deg > 180.0:
        deg -= 360.0
    while deg <= -180.0:
        deg += 360.0
    return deg

def _angle_diff_abs(a: float, b: float) -> float:
    return abs(_normalize180(a - b))

def _normalize180(deg: float) -> float:
    while deg > 180.0:
        deg -= 360.0
    while deg <= -180.0:
        deg += 360.0
    return deg

def _angdiff(a: float, b: float) -> float:
    return abs(_normalize180(a - b))

def _min_diff_to_set(angle: float, refs):
    return min(_angdiff(angle, r) for r in refs)

def set_perp_red_parallel_green(threshold_deg: float = 35.0, invert: bool = False):
    """
    Make the two approaches parallel to the vehicle heading GREEN
    and the two perpendicular approaches RED (frozen). Uses axis grouping so all
    4 sides are affected. If your map's light meshes are rotated 90°, set invert=True.
    """
    TLS = carla.TrafficLightState
    veh_yaw = vehicle.get_transform().rotation.yaw

    # Axis sets (cover both directions on an axis)
    parallel_axes = [veh_yaw, veh_yaw + 180.0]
    perp_axes     = [veh_yaw + 90.0, veh_yaw - 90.0]

    parallel, perpendicular, unsure = [], [], []
    lights = world.get_actors().filter('traffic.traffic_light')

    for tl in lights:
        try:
            lyaw = tl.get_transform().rotation.yaw

            d_par = _min_diff_to_set(lyaw, parallel_axes)
            d_per = _min_diff_to_set(lyaw, perp_axes)

            # Assign to the closest axis if within threshold; otherwise mark unsure
            if d_par <= threshold_deg or d_per <= threshold_deg:
                if d_par <= d_per:
                    parallel.append(tl)
                else:
                    perpendicular.append(tl)
            else:
                unsure.append(tl)  # rare, oddly oriented lights
        except Exception:
            pass

    # Optional flip if your map's light orientation seems 90° off
    if invert:
        parallel, perpendicular = perpendicular, parallel

    # Apply states + freeze
    # Parallel -> GREEN, Perp -> RED
    for tl in parallel:
        try:
            tl.set_state(TLS.Green)
            if _HAS_FREEZE:
                tl.set_time_is_frozen(True)
            else:
                tl.set_green_time(1e6); tl.set_yellow_time(0.001); tl.set_red_time(0.001)
        except Exception:
            pass

    for tl in perpendicular:
        try:
            tl.set_state(TLS.Red)
            if _HAS_FREEZE:
                tl.set_time_is_frozen(True)
            else:
                tl.set_red_time(1e6); tl.set_yellow_time(0.001); tl.set_green_time(0.001)
        except Exception:
            pass

    # Nudge "unsure" to the dominant group so they don’t stay out of sync
    if unsure:
        target_state = TLS.Green if len(parallel) >= len(perpendicular) else TLS.Red
        for tl in unsure:
            try:
                tl.set_state(target_state)
                if _HAS_FREEZE:
                    tl.set_time_is_frozen(True)
            except Exception:
                pass

    print(
        f"Parallel GREEN: {len(parallel)}, "
        f"Perpendicular RED: {len(perpendicular)}, "
        f"Unsure adjusted: {len(unsure)}"
    )

# === Teleport helper ===
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

# === Flip car helper ===
def flip_car_180():
    tf = vehicle.get_transform()
    tf.rotation.yaw = (tf.rotation.yaw + 180.0) % 360.0
    vehicle.set_transform(tf)
    print("Vehicle flipped 180 degrees.")

# ======================================================================
# =============== GAMEPAD (Xbox One) SUPPORT SECTION ====================
# ======================================================================
pygame.joystick.init()
_joystick = None
if pygame.joystick.get_count() > 0:
    _joystick = pygame.joystick.Joystick(0)
    _joystick.init()
    print(f"Gamepad connected: {_joystick.get_name()} | axes={_joystick.get_numaxes()} buttons={_joystick.get_numbuttons()} hats={_joystick.get_numhats()}")
else:
    print("No gamepad detected. Keyboard controls still available.")

# Keep previous button states for edge-detection
_prev_buttons = {}
_prev_hat = (0, 0)

# Heuristics for Xbox mapping (works across common SDL variants)
AXIS_LX = 0              # Left stick X
AXIS_LY = 1              # Unused here
AXIS_TRIGGERS_COMBINED = 2  # Some drivers map both triggers to one axis (-1..+1)
AXIS_RY = 3              # Unused here
AXIS_LT_MAYBE = 4        # Separate LT on some drivers (0..1 scaled from -1..+1)
AXIS_RT_MAYBE = 5        # Separate RT on some drivers

BTN_A, BTN_B, BTN_X, BTN_Y = 0, 1, 2, 3
BTN_LB, BTN_RB = 4, 5
BTN_BACK, BTN_START = 6, 7
BTN_LS, BTN_RS = 8, 9  # not used

def _deadzone(value: float, dz: float = 0.08) -> float:
    return 0.0 if abs(value) < dz else value

def _axis_or_zero(js, idx):
    try:
        return js.get_axis(idx)
    except Exception:
        return 0.0

def read_gamepad_controls():
    """
    Returns a dict with continuous controls and edge-triggered button presses.
    Continuous: steer [-1..1], throttle [0..1], brake [0..1], left_blinker(bool), right_blinker(bool)
    Edges: events list of strings describing actions to trigger once on button press.
    """
    global _prev_buttons, _prev_hat

    if _joystick is None:
        return dict(steer=0.0, throttle=0.0, brake=0.0,
                    left_blinker=False, right_blinker=False, events=[])

    # --- Steering from left stick X ---
    steer = _deadzone(_axis_or_zero(_joystick, AXIS_LX))

    # --- Triggers (various driver mappings) ---
    # Preferred: separate LT/RT at indices 4/5 (reported -1..+1; convert to 0..1)
    lt = rt = 0.0
    axes = _joystick.get_numaxes()
    if axes >= 6:
        lt = max(0.0, (_axis_or_zero(_joystick, AXIS_LT_MAYBE) + 1.0) * 0.5)
        rt = max(0.0, (_axis_or_zero(_joystick, AXIS_RT_MAYBE) + 1.0) * 0.5)
    else:
        # Fallback: combined triggers axis at index 2
        comb = _axis_or_zero(_joystick, AXIS_TRIGGERS_COMBINED)
        # Many drivers: rest ~ -1; LT pushes towards 0..+1, RT pushes towards 0..+1 but cannot be read separately.
        # Heuristic split: treat positive as RT, negative magnitude as LT.
        rt = max(0.0, comb)  # 0..+1
        lt = max(0.0, -comb) # 0..+1

    # Deadzone/smoothing
    lt = 0.0 if lt < 0.02 else lt
    rt = 0.0 if rt < 0.02 else rt

    # Indicators from LB/RB (hold = on)
    left_blinker = False
    right_blinker = False
    try:
        left_blinker  = _joystick.get_button(BTN_LB) == 1
        right_blinker = _joystick.get_button(BTN_RB) == 1
    except Exception:
        pass

    # --- Edge-triggered button presses (A/B/X/Y/Back) ---
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
    # NEW: L1 / LB edge -> perpendicular RED, parallel GREEN
    if pressed(BTN_LB):
        events.append("PERP_RED_PAR_GREEN")

    # Hat (D-pad) not used here but we track to keep prev state synced
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

# ======================================================================

# === Clock ===
clock = pygame.time.Clock()

print("Controls:")
print("Keyboard — W: throttle, S: brake/reverse, A/D: steer, Q/E: turn signals")
print("Keyboard — T: teleport, V: OFF + capture, R: RED + capture, G: GREEN + capture, F: flip 180°, ESC: quit")
print("Gamepad (Xbox One) — Left Stick X: steer, RT: throttle, LT: brake, LB/RB: turn signals (hold)")
print("Gamepad — A: GREEN + capture, B: RED + capture, X: OFF/NO LIGHT + capture, Y: flip 180°, Back: teleport")
print("Gamepad — LB (tap): PERP RED / PAR GREEN (freeze)")

# === Main loop ===
try:
    while True:
        clock.tick(30)

        # Must pump events to keep joystick state fresh
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
                    start_capture(label_value=0)      # 0 = red
                elif event.key == pygame.K_g:
                    force_all_lights(TLS.Green)
                    start_capture(label_value=1)      # 1 = green
                elif event.key == pygame.K_v:
                    force_all_lights_off()
                    start_capture(label_value=-1)     # -1 = no light
                elif event.key == pygame.K_f:
                    flip_car_180()

        # Read keyboard state (continuous)
        keys = pygame.key.get_pressed()

        # Read controller state (continuous + edge events)
        gp = read_gamepad_controls()

        control = carla.VehicleControl()

        # Current speed to decide reverse behavior on keyboard 'S'
        vel = vehicle.get_velocity()
        speed = (vel.x**2 + vel.y**2 + vel.z**2) ** 0.5

        # --- Throttle / Brake merge (keyboard vs gamepad) ---
        # Keyboard
        kb_throttle = 0.6 if keys[pygame.K_w] else 0.0
        if keys[pygame.K_s]:
            # Keyboard S: brake strongly if moving, else reverse hard
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

        # Gamepad
        gp_throttle = float(gp["throttle"])  # 0..1
        gp_brake = float(gp["brake"])        # 0..1

        # Merge strategy: choose the stronger input in each category
        throttle = max(kb_throttle, gp_throttle)
        brake = max(kb_brake, gp_brake)

        # Steering merge
        steer = 0.0
        if keys[pygame.K_a]:
            steer -= 0.5
        if keys[pygame.K_d]:
            steer += 0.5
        # Gamepad steering (scaled to similar range, with light damping)
        steer_gp = gp["steer"] * 0.7
        # Prefer explicit keyboard steer if pressed; otherwise use gamepad
        if not (keys[pygame.K_a] or keys[pygame.K_d]):
            steer = steer_gp

        # Indicators
        control.left_indicator_light = keys[pygame.K_q] or gp["left_blinker"]
        control.right_indicator_light = keys[pygame.K_e] or gp["right_blinker"]

        # Apply reverse logic (keyboard only, as requested)
        control.throttle = throttle
        control.brake = brake
        control.steer = np.clip(steer, -1.0, 1.0)
        control.reverse = kb_reverse
        if kb_reverse:
            control.throttle = kb_rev_throttle  # override when reversing via keyboard

        vehicle.apply_control(control)

        # Spectator follow-cam
        transform = vehicle.get_transform()
        forward_vector = transform.get_forward_vector()
        cam_location = transform.location - forward_vector * 8 + carla.Location(z=3)
        cam_rotation = carla.Rotation(pitch=-10, yaw=transform.rotation.yaw)
        spectator.set_transform(carla.Transform(cam_location, cam_rotation))

        # Draw camera
        if camera_image_surface:
            screen.blit(camera_image_surface, (0, 0))
            pygame.display.flip()

        # Handle edge-triggered gamepad events AFTER apply_control so they don't block driving
        for evt in gp["events"]:
            if evt == "GREEN_CAPTURE":
                # force_all_lights(TLS.Green)
                set_perp_red_parallel_green(invert=True)
                start_capture(label_value=1)
            elif evt == "RED_CAPTURE":
                # force_all_lights(TLS.Red)
                set_perp_red_parallel_green()
                start_capture(label_value=0)
            elif evt == "NO_LIGHT_CAPTURE":
                # force_all_lights_off()
                start_capture(label_value=-1)
            elif evt == "TELEPORT":
                teleport_to_intersection()
            elif evt == "FLIP_180":
                flip_car_180()
            elif evt == "PERP_RED_PAR_GREEN":
                pass
                # set_perp_red_parallel_green()

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
