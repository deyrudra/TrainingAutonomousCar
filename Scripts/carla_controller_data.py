import os
import math
import time
import random
import threading
import numpy as np
import pygame
import carla
from carla import VehicleLightState

WINDOW_W, WINDOW_H = 320, 240
FPS = 60

# Camera
CAM_W, CAM_H = 320, 240
FOV = 120
SENSOR_TICK = 0.1  

# Traffic
NUM_NPC_VEHICLES = 30
TRAFFIC_MANAGER_PORT = 8000
GLOBAL_LEAD_DIST = 2.5
SEED_TRAFFIC = 42

# Output
OUT_DIR = r"C:/carla_data/dataset_run_manual"

USE_GAMEPAD = True

AXIS_LX = 0                
AXIS_L2 = 4                 
AXIS_R2 = 5             

# Controller buttons on ps5 
BTN_CROSS    = 0  
BTN_CIRCLE   = 1  
BTN_SQUARE   = 2  
BTN_TRIANGLE = 3  
BTN_OPTIONS  = 4  
BTN_L1 = 9
BTN_R1 = 10

# Turn signals on the controller
BTN_LEFT_SIGNAL   = BTN_L1
BTN_RIGHT_SIGNAL  = BTN_R1
BTN_CANCEL_SIGNAL = BTN_CIRCLE
BTN_RESPAWN       = BTN_OPTIONS

STEER_DEADZONE = 0.08
STEER_SMOOTH = 0.3
TRIGGER_DEADZONE = 0.05
KEY_STEER_STEP = 0.05

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def speed_ms(vehicle):
    v = vehicle.get_velocity()
    return math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z)

def follow_spectator(get_vehicle_fn, world, stop_event):
    spectator = world.get_spectator()
    back = 6.0
    height = 3.0
    while not stop_event.is_set():
        v = get_vehicle_fn()
        if v is None:
            time.sleep(0.03)
            continue
        try:
            tf = v.get_transform()
        except RuntimeError:
            time.sleep(0.03)
            continue

        yaw = math.radians(tf.rotation.yaw)
        offx = -back * math.cos(yaw)
        offy = -back * math.sin(yaw)
        cam_loc = carla.Location(
            x=tf.location.x + offx,
            y=tf.location.y + offy,
            z=tf.location.z + height
        )
        cam_rot = carla.Rotation(pitch=-10, yaw=tf.rotation.yaw)
        spectator.set_transform(carla.Transform(cam_loc, cam_rot))
        time.sleep(0.03)

def spawn_traffic(client, world, bp_lib, ego_vehicle, num=30):
    tm = client.get_trafficmanager(TRAFFIC_MANAGER_PORT)
    tm.set_global_distance_to_leading_vehicle(GLOBAL_LEAD_DIST)
    tm.set_synchronous_mode(False)
    tm.set_random_device_seed(SEED_TRAFFIC)

    sps = world.get_map().get_spawn_points()
    random.shuffle(sps)
    ego_loc = ego_vehicle.get_transform().location
    sps = [sp for sp in sps if sp.location.distance(ego_loc) > 8.0]

    vbps = bp_lib.filter('vehicle.*')
    batch = []
    used = 0
    for sp in sps:
        if used >= num:
            break
        bp = random.choice(vbps)
        if bp.has_attribute('color'):
            bp.set_attribute('color', random.choice(bp.get_attribute('color').recommended_values))
        batch.append(carla.command.SpawnActor(bp, sp))
        used += 1

    spawned = []
    if batch:
        resp = client.apply_batch_sync(batch, True)
        actor_ids = [r.actor_id for r in resp if not r.error]
        if actor_ids:
            client.apply_batch_sync(
                [carla.command.SetAutopilot(aid, True, tm.get_port()) for aid in actor_ids],
                True
            )
            for aid in actor_ids:
                a = world.get_actor(aid)
                if a:
                    spawned.append(a)
    print(f"[TRAFFIC] Spawned {len(spawned)} NPC vehicles.")
    return spawned

def relock_spectator_to(world, v):
    spectator = world.get_spectator()
    tf = v.get_transform()
    fwd = tf.get_forward_vector()
    cam_loc = tf.location - fwd * 6 + carla.Location(z=3)
    cam_rot = carla.Rotation(pitch=-10, yaw=tf.rotation.yaw)
    spectator.set_transform(carla.Transform(cam_loc, cam_rot))

def lightstate_to_signal(ls: VehicleLightState) -> int:
    if ls & VehicleLightState.LeftBlinker:
        return -1
    if ls & VehicleLightState.RightBlinker:
        return 1
    return 0

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def apply_deadzone(x, dz):
    return 0.0 if abs(x) < dz else x

def normalize_trigger(axis_val):
   
    return clamp((axis_val + 1.0) * 0.5, 0.0, 1.0)

def main():
    ensure_dir(OUT_DIR)
    print(f"[RUN] Saving to: {OUT_DIR}")

    pygame.init()
    pygame.joystick.init()

    joystick = None
    if USE_GAMEPAD and pygame.joystick.get_count() > 0:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        print(f"[GAMEPAD] Detected: {joystick.get_name()} (axes={joystick.get_numaxes()}, buttons={joystick.get_numbuttons()})")
    else:
        print("[GAMEPAD] None detected (keyboard controls enabled).")

    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("CARLA Manual Drive + Logger (320x240, m/s, PS5 controller)")
    clock = pygame.time.Clock()

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()

    # spawn the tesla 
    vehicle_bp = bp_lib.find('vehicle.tesla.model3')
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        print("No spawn points found!")
        return

    def try_spawn():
        random.shuffle(spawn_points)
        for sp in spawn_points:
            v = world.try_spawn_actor(vehicle_bp, sp)
            if v:
                return v
        return world.spawn_actor(vehicle_bp, random.choice(spawn_points))

    vehicle = try_spawn()
    print(f"[EGO] Spawned: {vehicle.type_id}")

    # Camera
    cam_bp = bp_lib.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', str(CAM_W))
    cam_bp.set_attribute('image_size_y', str(CAM_H))
    cam_bp.set_attribute('fov', str(FOV))
    cam_bp.set_attribute('sensor_tick', str(SENSOR_TICK))
    cam_transform = carla.Transform(carla.Location(x=1.5, z=2.2))
    camera = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)

    images = []
    angles = []
    speeds_ms = []
    turn_signals = []

    camera_surface = None

    def on_image(image):
        nonlocal camera_surface
        arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((CAM_H, CAM_W, 4))
        bgr = arr[:, :, :3]
        rgb = bgr[:, :, ::-1] 
        camera_surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))

        ctrl = vehicle.get_control()
        ls = vehicle.get_light_state()

        images.append(rgb.copy())                             
        angles.append(np.float32(ctrl.steer))                 
        speeds_ms.append(np.float32(speed_ms(vehicle)))        
        turn_signals.append(np.int8(lightstate_to_signal(ls)))

    camera.listen(on_image)

    # camera following (spectator)
    current = {"vehicle": vehicle}
    stop_spec = threading.Event()
    spec_thread = threading.Thread(
        target=follow_spectator,
        args=(lambda: current["vehicle"], world, stop_spec),
        daemon=True
    )
    spec_thread.start()

    # spawn the traffic (others cars on the map)
    npc_vehicles = spawn_traffic(client, world, bp_lib, vehicle, NUM_NPC_VEHICLES)

    # Controls
    control = carla.VehicleControl()
    steer_cmd = 0.0

    last_left_gp = False
    last_right_gp = False
    last_left_key = False
    last_right_key = False

    #spawns in a new tesla 
    def respawn_ego():
        nonlocal camera, vehicle, camera_surface, control, steer_cmd
        try:
            if camera.is_listening:
                camera.stop()
        except:
            pass
        try:
            camera.destroy()
        except:
            pass
        try:
            vehicle.destroy()
        except:
            pass

        v = try_spawn()
        vehicle = v
        current["vehicle"] = v
        vehicle.set_light_state(VehicleLightState.NONE)
        control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0)
        steer_cmd = 0.0

        camera = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)
        camera.listen(on_image)
        relock_spectator_to(world, vehicle)
        print("[EGO] Respawned.")

    can_respawn = True

    try:
        print("Controls:")
        print("  Gamepad: LX steer, R2 throttle, L2 brake, L1 toggle left, R1 toggle right, O cancel, Options respawn")
        print("  Keyboard: Left/Right steer, Up throttle, Space brake, A toggle left, D toggle right, R cancel, L respawn, ESC quit")

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.JOYDEVICEADDED and USE_GAMEPAD and joystick is None:
                    j = pygame.joystick.Joystick(event.device_index)
                    j.init()
                    joystick = j
                    print(f"[GAMEPAD] Connected: {joystick.get_name()} (axes={joystick.get_numaxes()}, buttons={joystick.get_numbuttons()})")
                elif event.type == pygame.JOYDEVICEREMOVED and joystick is not None:
                    print("[GAMEPAD] Disconnected.")
                    joystick = None
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_l and can_respawn:
                        can_respawn = False
                        respawn_ego()
                    elif event.key == pygame.K_ESCAPE:
                        running = False
                elif event.type == pygame.KEYUP:
                    if event.key == pygame.K_l:
                        can_respawn = True

            
            keys = pygame.key.get_pressed()

            # keyboard controls
            steer_target = 0.0
            throttle_target = 1.0 if keys[pygame.K_UP] else 0.0
            brake_target = 1.0 if keys[pygame.K_SPACE] else 0.0

            if keys[pygame.K_LEFT]:
                steer_target = clamp(steer_cmd - KEY_STEER_STEP, -1.0, 1.0)
            elif keys[pygame.K_RIGHT]:
                steer_target = clamp(steer_cmd + KEY_STEER_STEP, -1.0, 1.0)
            else:
                steer_target = 0.0

            if USE_GAMEPAD and joystick is not None:
                try:
                    lx = joystick.get_axis(AXIS_LX)
                    lx = apply_deadzone(lx, STEER_DEADZONE)
                    steer_target = lx

                    l2_raw = joystick.get_axis(AXIS_L2)
                    r2_raw = joystick.get_axis(AXIS_R2)
                    l2 = normalize_trigger(l2_raw)
                    r2 = normalize_trigger(r2_raw)

                    throttle_target = 0.0 if r2 < TRIGGER_DEADZONE else r2
                    brake_target = 0.0 if l2 < TRIGGER_DEADZONE else l2

                    #turn signal is pressed (on/off)
                    left_pressed = bool(joystick.get_button(BTN_LEFT_SIGNAL))
                    right_pressed = bool(joystick.get_button(BTN_RIGHT_SIGNAL))
                    cancel_pressed = bool(joystick.get_button(BTN_CANCEL_SIGNAL))

                    if left_pressed and not last_left_gp:
                        if vehicle.get_light_state() & VehicleLightState.LeftBlinker:
                            vehicle.set_light_state(VehicleLightState.NONE)
                        else:
                            vehicle.set_light_state(VehicleLightState.LeftBlinker)
                    if right_pressed and not last_right_gp:
                        if vehicle.get_light_state() & VehicleLightState.RightBlinker:
                            vehicle.set_light_state(VehicleLightState.NONE)
                        else:
                            vehicle.set_light_state(VehicleLightState.RightBlinker)
                    if cancel_pressed:
                        vehicle.set_light_state(VehicleLightState.NONE)

                    last_left_gp = left_pressed
                    last_right_gp = right_pressed

                    # respawn when pressing options 
                    if joystick.get_button(BTN_RESPAWN) and can_respawn:
                        can_respawn = False
                        respawn_ego()
                    if joystick.get_button(BTN_RESPAWN) == 0:
                        can_respawn = True

                except Exception:
                    pass

           #turn signal using keyboard controls 
            left_key_pressed = keys[pygame.K_a]
            right_key_pressed = keys[pygame.K_d]
            cancel_key_pressed = keys[pygame.K_r]

            if left_key_pressed and not last_left_key:
                if vehicle.get_light_state() & VehicleLightState.LeftBlinker:
                    vehicle.set_light_state(VehicleLightState.NONE)
                else:
                    vehicle.set_light_state(VehicleLightState.LeftBlinker)
            if right_key_pressed and not last_right_key:
                if vehicle.get_light_state() & VehicleLightState.RightBlinker:
                    vehicle.set_light_state(VehicleLightState.NONE)
                else:
                    vehicle.set_light_state(VehicleLightState.RightBlinker)
            if cancel_key_pressed:
                vehicle.set_light_state(VehicleLightState.NONE)

            last_left_key = left_key_pressed
            last_right_key = right_key_pressed

            steer_cmd = steer_cmd + (steer_target - steer_cmd) * STEER_SMOOTH
            control.steer = clamp(steer_cmd, -1.0, 1.0)

            control.throttle = clamp(throttle_target, 0.0, 1.0)
            control.brake = clamp(brake_target, 0.0, 1.0)
            control.reverse = False

            vehicle.apply_control(control)

            if camera_surface is not None:
                screen.blit(camera_surface, (0, 0))
                font = pygame.font.SysFont(None, 18)
                spd = speed_ms(vehicle)
                sig = lightstate_to_signal(vehicle.get_light_state())
                hud1 = font.render(f"Steer: {control.steer:+.2f}", True, (255, 255, 0))
                hud2 = font.render(f"Throttle:{control.throttle:.2f} Brake:{control.brake:.2f}", True, (200, 255, 255))
                hud3 = font.render(f"Speed: {spd:.2f} m/s  Signal: {sig}", True, (255, 200, 0))
                hud4 = font.render("Options: Respawn  ESC: Quit", True, (255, 255, 255))
                screen.blit(hud1, (6, 6))
                screen.blit(hud2, (6, 24))
                screen.blit(hud3, (6, 42))
                screen.blit(hud4, (6, 60))
                pygame.display.flip()

            clock.tick(FPS)

    except KeyboardInterrupt:
        pass
    finally:
        print(f"[SAVE] Saving {len(images)} frames to {OUT_DIR} ...")

        try:
            stop_spec.set()
            spec_thread.join(timeout=0.5)
        except:
            pass

        try:
            if camera.is_listening:
                camera.stop()
        except:
            pass

        try:
            np.save(os.path.join(OUT_DIR, "images.npy"), np.array(images, dtype=np.uint8))
            np.save(os.path.join(OUT_DIR, "angles.npy"), np.array(angles, dtype=np.float32))
            np.save(os.path.join(OUT_DIR, "speeds_ms.npy"), np.array(speeds_ms, dtype=np.float32))
            np.save(os.path.join(OUT_DIR, "turn_signals.npy"), np.array(turn_signals, dtype=np.int8))
            print("[SAVE] Done.")
        except Exception as e:
            print(f"[SAVE] FAILED: {e}")

        try:
            camera.destroy()
        except:
            pass
        try:
            vehicle.destroy()
        except:
            pass
        try:
            for a in npc_vehicles:
                try:
                    if a and a.is_alive:
                        a.destroy()
                except:
                    pass
        except:
            pass

        pygame.quit()
        print("[DONE] All data saved & cleaned up.")


if __name__ == "__main__":
    main()
