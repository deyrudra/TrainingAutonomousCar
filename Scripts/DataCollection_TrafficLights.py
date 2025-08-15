# this script lets us drive a carla vehicle by keyboard or xbox gamepad. we mirror a spectator camera behind the car.
# we also capture cropped frames around traffic lights into a numpy dataset for later training.
# the capture can be triggered for red, green, or no-light classes. we try to keep the flow simple and responsive.

import carla
import pygame
import time
import random
import numpy as np
import os
import cv2  # used to resize frames to 224x224 for the dataset

# pygame window. simple 1024x768 view. we draw the rgb camera to this surface.
pygame.init()
width, height = 1024, 768
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("carla manual control + tl capture")

# connect to carla on localhost. we load town05 and pull the world, blueprints, and map.
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)
client.load_world("Town05")
world = client.get_world()
blueprint_library = world.get_blueprint_library()
map_ = world.get_map()

# here we wire up the traffic manager so that background npcs can drive around us.
# we keep it synchronous to reduce odd timing glitches during data capture.
tm = client.get_trafficmanager()
tm_port = tm.get_port()
tm.set_synchronous_mode(True)

# we spawn a bunch of background vehicles with random speed modifiers.
# this adds variety to scenes. toggle the loop size or comment out if we want a quiet map.
background_vehicles = []
for _ in range(50):
    npc_bp = random.choice(blueprint_library.filter('vehicle.*'))
    npc_spawn = random.choice(world.get_map().get_spawn_points())
    npc = world.try_spawn_actor(npc_bp, npc_spawn)
    if npc:
        npc.set_autopilot(True, tm_port)
        tm.vehicle_percentage_speed_difference(npc, random.randint(0, 30))
        background_vehicles.append(npc)

# this spawns our controllable vehicle. it tries several spawn points to avoid collisions.
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

# we keep a spectator camera following behind the car so it is easier to see context.
spectator = world.get_spectator()

# attach a single rgb camera to the hood. settings match the window so blits are fast.
def _attach_camera(world, vehicle):
    bp_lib = world.get_blueprint_library()
    camera_bp = bp_lib.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '1024')
    camera_bp.set_attribute('image_size_y', '768')
    camera_bp.set_attribute('fov', '110')
    camera_bp.set_attribute('sensor_tick', '0.1')  # roughly 10 fps, good for manual driving + capture
    cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    cam = world.spawn_actor(camera_bp, cam_transform, attach_to=vehicle)
    return cam

camera = _attach_camera(world, vehicle)

# camera frame state. we hold the latest rgb frame as both numpy and pygame surface.
camera_image_surface = None
camera_frame_rgb_native = None

# capture state. we snapshot a short burst of frames each time we trigger.
# labels are -1 (no light), 0 (red), 1 (green). bursts reduce label jitter and give us mini sequences.
CAPTURE_FRAMES = 7
capture_active = False
capture_remaining = 0
capture_label_value = None
capture_buffer_images = []
capture_buffer_labels = []

# output paths. we append to these .npy files if they already exist. each save call stacks the new chunk.
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

    counts = {label: int(np.sum(lbls == label)) for label in [-1, 0, 1]}
    print(f"saved {len(new_lbls)} frames.")
    print(f"totals -> images: {imgs.shape[0]}, labels: {lbls.shape[0]}")
    print(f"label counts: no light (-1): {counts[-1]}, red (0): {counts[0]}, green (1): {counts[1]}")

def start_capture(label_value):
    # this arms the capture for the next few frames. we keep it short to avoid too many near-duplicates.
    global capture_active, capture_remaining, capture_label_value
    if capture_active:
        print("capture in progress. pls wait.")
        return
    capture_active = True
    capture_remaining = CAPTURE_FRAMES
    capture_label_value = label_value
    capture_buffer_images.clear()
    capture_buffer_labels.clear()
    print(f"capture started for label {label_value} (next {CAPTURE_FRAMES} frames).")

# this crop brings attention to the tl area: center zoom, blackout lower third, and blur side strips.
# it helps models ignore road texture and car hood clutter. we keep color so red/green hue is intact.
def traffic_light_crop(img: np.ndarray) -> np.ndarray:
    if img is None or img.ndim != 3:
        return img

    h, w = img.shape[:2]
    out = img

    # center zoom
    z = 1.0 + (50.0 / 100.0)
    new_w = max(1, int(w / z))
    new_h = max(1, int(h / z))
    x1 = max(0, (w - new_w) // 2)
    y1 = max(0, (h - new_h) // 2)
    cropped = out[y1:y1 + new_h, x1:x1 + new_w]
    if cropped.size > 0:
        out = cv2.resize(cropped, (w, h))

    # blackout bottom third
    out[h * 2 // 3 :, ...] = 0

    # soft blur on left/right quarters
    quarter_w = max(1, w // 4)

    def blur_edge(region: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(region, (21, 21), 0)

    out[:, :quarter_w, :] = blur_edge(out[:, :quarter_w, :])
    out[:, -quarter_w:, :] = blur_edge(out[:, -quarter_w:, :])

    return out

# this converts raw sensor bytes to an rgb numpy frame and pygame surface.
# when armed, it also downsamples to 224x224 and buffers into the current capture chunk.
def process_image(image):
    global camera_image_surface, camera_frame_rgb_native, capture_active, capture_remaining
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))[:, :, :3][:, :, ::-1]  # bgra->bgr->rgb

    array = traffic_light_crop(array)

    camera_frame_rgb_native = array
    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    camera_image_surface = pygame.transform.scale(surface, (width, height))

    if capture_active and capture_remaining > 0:
        resized_img = cv2.resize(camera_frame_rgb_native, (224, 224))
        capture_buffer_images.append(resized_img.copy())
        capture_buffer_labels.append(capture_label_value)
        capture_remaining -= 1
        print(f"captured frame {CAPTURE_FRAMES - capture_remaining}/{CAPTURE_FRAMES} (label={capture_label_value})")
        if capture_remaining == 0:
            save_dataset(capture_buffer_images, capture_buffer_labels)
            print("capture finished.")
            capture_active = False

camera.listen(process_image)

# helpers to steer traffic lights for fast labeling. we try to freeze states when the api supports it.
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
    print(f"forced {count} traffic lights to {state.name.lower()}.")

def force_all_lights_off():
    # some builds support a true off state. if not, we unfreeze and reset timings to normal cycling.
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
        print(f"set {turned_off} traffic lights to off.")
    if fallback:
        print(f"off not supported here. reverted {fallback} to normal timing.")

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

# these helpers align lights relative to our vehicle yaw and set parallel vs perpendicular axes.
# this speeds up labeling intersections without hunting for a specific signal asset.
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
    make approaches parallel to the vehicle heading green. make perpendicular approaches red.
    we freeze states. if the map meshes are rotated 90°, set invert=True.
    """
    TLS = carla.TrafficLightState
    veh_yaw = vehicle.get_transform().rotation.yaw

    parallel_axes = [veh_yaw, veh_yaw + 180.0]
    perp_axes     = [veh_yaw + 90.0, veh_yaw - 90.0]

    parallel, perpendicular, unsure = [], [], []
    lights = world.get_actors().filter('traffic.traffic_light')

    for tl in lights:
        try:
            lyaw = tl.get_transform().rotation.yaw
            d_par = _min_diff_to_set(lyaw, parallel_axes)
            d_per = _min_diff_to_set(lyaw, perp_axes)
            if d_par <= threshold_deg or d_per <= threshold_deg:
                if d_par <= d_per:
                    parallel.append(tl)
                else:
                    perpendicular.append(tl)
            else:
                unsure.append(tl)
        except Exception:
            pass

    if invert:
        parallel, perpendicular = perpendicular, parallel

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
        f"parallel green: {len(parallel)}, "
        f"perpendicular red: {len(perpendicular)}, "
        f"unsure adjusted: {len(unsure)}"
    )

# teleport helps us hop to a random intersection. handy for quick dataset variety.
def teleport_to_intersection():
    waypoints = [wp for wp in map_.generate_waypoints(2.0) if wp.is_junction]
    if not waypoints:
        print("no intersections found.")
        return
    wp = random.choice(waypoints)
    tf = wp.transform
    tf.location.z += 0.5
    vehicle.set_transform(tf)
    vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
    vehicle.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
    print(f"teleported near junction at {tf.location}")

# sometimes we need to flip around fast to face the opposite axis. this does a 180 on yaw.
def flip_car_180():
    tf = vehicle.get_transform()
    tf.rotation.yaw = (tf.rotation.yaw + 180.0) % 360.0
    vehicle.set_transform(tf)
    print("vehicle flipped 180 degrees.")

# gamepad support. we try to be robust to sdl mappings. keyboard still works fine in parallel.
pygame.joystick.init()
_joystick = None
if pygame.joystick.get_count() > 0:
    _joystick = pygame.joystick.Joystick(0)
    _joystick.init()
    print(f"gamepad: {_joystick.get_name()} | axes={_joystick.get_numaxes()} buttons={_joystick.get_numbuttons()} hats={_joystick.get_numhats()}")
else:
    print("no gamepad detected. keyboard controls still work.")

_prev_buttons = {}
_prev_hat = (0, 0)

# typical xbox layout on many drivers. some variance exists. our code is defensive.
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

# read the current gamepad state. we return continuous controls plus a list of edge events.
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
    if pressed(BTN_LB):
        events.append("PERP_RED_PAR_GREEN")

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

# steady clock. 30 fps feels right for human input without being too heavy.
clock = pygame.time.Clock()

print("controls:")
print("keyboard — w throttle, s brake/reverse, a/d steer, q/e turn signals")
print("keyboard — t teleport, v off+capture, r red+capture, g green+capture, f flip 180°, esc quit")
print("gamepad — left x steer, rt throttle, lt brake, lb/rb signals (hold)")
print("gamepad — a green+cap, b red+cap, x off/no-light+cap, y flip, back teleport, lb tap align tls")

# main loop. we read input, apply control, update spectator, draw the hud, and process label events.
try:
    while True:
        clock.tick(30)

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
                    start_capture(label_value=0)      # red
                elif event.key == pygame.K_g:
                    force_all_lights(TLS.Green)
                    start_capture(label_value=1)      # green
                elif event.key == pygame.K_v:
                    force_all_lights_off()
                    start_capture(label_value=-1)     # no light
                elif event.key == pygame.K_f:
                    flip_car_180()

        keys = pygame.key.get_pressed()
        gp = read_gamepad_controls()

        control = carla.VehicleControl()

        vel = vehicle.get_velocity()
        speed = (vel.x**2 + vel.y**2 + vel.z**2) ** 0.5

        # merge keyboard and gamepad throttle/brake. choose the stronger input in each category.
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

        gp_throttle = float(gp["throttle"])
        gp_brake = float(gp["brake"])

        throttle = max(kb_throttle, gp_throttle)
        brake = max(kb_brake, gp_brake)

        # merge steering. keyboard has priority if pressed; else use gamepad with light damping.
        steer = 0.0
        if keys[pygame.K_a]:
            steer -= 0.5
        if keys[pygame.K_d]:
            steer += 0.5
        steer_gp = gp["steer"] * 0.7
        if not (keys[pygame.K_a] or keys[pygame.K_d]):
            steer = steer_gp

        # indicator lights reflect keyboard q/e or gamepad lb/rb
        control.left_indicator_light = keys[pygame.K_q] or gp["left_blinker"]
        control.right_indicator_light = keys[pygame.K_e] or gp["right_blinker"]

        # apply reverse if we are holding s at standstill. this keeps behavior intuitive.
        control.throttle = throttle
        control.brake = brake
        control.steer = np.clip(steer, -1.0, 1.0)
        control.reverse = kb_reverse
        if kb_reverse:
            control.throttle = kb_rev_throttle

        vehicle.apply_control(control)

        # keep spectator a few meters back and slightly above. gentle chase cam.
        transform = vehicle.get_transform()
        forward_vector = transform.get_forward_vector()
        cam_location = transform.location - forward_vector * 8 + carla.Location(z=3)
        cam_rotation = carla.Rotation(pitch=-10, yaw=transform.rotation.yaw)
        spectator.set_transform(carla.Transform(cam_location, cam_rotation))

        # draw the latest camera frame
        if camera_image_surface:
            screen.blit(camera_image_surface, (0, 0))
            pygame.display.flip()

        # handle one-shot gamepad events after control so we do not block the driving loop
        for evt in gp["events"]:
            if evt == "GREEN_CAPTURE":
                set_perp_red_parallel_green(invert=True)
                start_capture(label_value=1)
            elif evt == "RED_CAPTURE":
                set_perp_red_parallel_green()
                start_capture(label_value=0)
            elif evt == "NO_LIGHT_CAPTURE":
                start_capture(label_value=-1)
            elif evt == "TELEPORT":
                teleport_to_intersection()
            elif evt == "FLIP_180":
                flip_car_180()
            elif evt == "PERP_RED_PAR_GREEN":
                pass
                # set_perp_red_parallel_green()

except KeyboardInterrupt:
    print("exiting and cleaning up...")

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
    print("vehicle and camera destroyed. traffic lights restored. goodbye.")
