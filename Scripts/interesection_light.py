import os
import math
import time
import random
import threading
import atexit
import numpy as np
import pygame
import carla


OUT_DIR = r"C:/carla_data/traffic_lights_run_2"  
TARGET_COUNT = 100                
CAM_W, CAM_H = 320, 240
FOV = 90
SENSOR_TICK = 0.05
WINDOW_W, WINDOW_H = CAM_W, CAM_H
FPS = 60
USE_GAMEPAD = True               
AUTOSAVE_EVERY = 1                

#controller movement using L3, R3
AXIS_LX = 0
AXIS_L2 = 4
AXIS_R2 = 5

KEEP_ALIVE = []
G_IMAGES = []
G_LABELS = []
G_OUTDIR = OUT_DIR

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def apply_deadzone(x, dz):
    return 0.0 if abs(x) < dz else x

def normalize_trigger(axis_val):
    return clamp((axis_val + 1.0) * 0.5, 0.0, 1.0)

def follow_spectator(get_vehicle_fn, get_world_fn, stop_event):
    # current tesla in third person
    back = 6.0
    height = 3.0
    while not stop_event.is_set():
        v = get_vehicle_fn()
        w = get_world_fn()
        if v is None or w is None:
            time.sleep(0.05)
            continue
        try:
            tf = v.get_transform()
            spectator = w.get_spectator()
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
        except Exception:
            pass
        time.sleep(0.03)


def create_camera(world, bp, transform, parent, on_image_cb):
    cam = world.spawn_actor(bp, transform, attach_to=parent)
    cam.listen(on_image_cb)
    KEEP_ALIVE.append(cam)  
    return cam

def safe_stop_destroy_sensor(sensor):
    if not sensor:
        return
    try:
        sensor.listen(None)    
    except:
        pass
    try:
        if getattr(sensor, "is_listening", False):
            sensor.stop()
    except:
        pass
    try:
        sensor.destroy()
    except:
        pass

# needed to save, because CARLA kept crashing 
def save_atomic(out_dir, images, labels):
    ensure_dir(out_dir)
    # changed file names to 2nd, so I can add them to the original 
    final_imgs = os.path.join(out_dir, "intersectionlight2.npy")
    final_lbls = os.path.join(out_dir, "labels2.npy")
    part_imgs  = final_imgs + ".part"
    part_lbls  = final_lbls + ".part"

    with open(part_imgs, "wb") as f:
        np.save(f, np.array(images, dtype=np.uint8))
        f.flush(); os.fsync(f.fileno())
    with open(part_lbls, "wb") as f:
        np.save(f, np.array(labels, dtype=np.int8))
        f.flush(); os.fsync(f.fileno())

    os.replace(part_imgs, final_imgs)
    os.replace(part_lbls, final_lbls)

    print(f"[AUTOSAVE] Saved {len(labels)} samples -> {final_imgs}, {final_lbls}")

def atexit_handler():
    try:
        if G_LABELS: 
            save_atomic(G_OUTDIR, G_IMAGES, G_LABELS)
            print("[ATEXIT] Final autosave complete.")
    except Exception as e:
        print(f"[ATEXIT] Save failed: {e}")

atexit.register(atexit_handler)

def main():
    global G_IMAGES, G_LABELS
    ensure_dir(OUT_DIR)
    print(f"[RUN] Saving to: {OUT_DIR}")

    pygame.init()
    pygame.joystick.init()

    joystick = None
    if USE_GAMEPAD and pygame.joystick.get_count() > 0:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        print(f"[GAMEPAD] {joystick.get_name()} (axes={joystick.get_numaxes()}, buttons={joystick.get_numbuttons()})")

    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Traffic Light Collector (320x240)")
    clock = pygame.time.Clock()

    current = {"client": None, "world": None, "bp": None, "vehicle": None, "camera": None}

    def connect_world():
        c = carla.Client("localhost", 2000)
        c.set_timeout(10.0)
        w = c.get_world()
        bp = w.get_blueprint_library()
        return c, w, bp

    def try_spawn(w, bp, vehicle_bp, spawn_points):
        random.shuffle(spawn_points)
        for sp in spawn_points:
            v = w.try_spawn_actor(vehicle_bp, sp)
            if v:
                return v
        return w.spawn_actor(vehicle_bp, random.choice(spawn_points))

    
    client, world, bp = connect_world()
    current["client"], current["world"], current["bp"] = client, world, bp

    # Spawn Vehicle 
    vehicle_bp = bp.find("vehicle.tesla.model3")
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        print("No spawn points found.")
        return

    vehicle = try_spawn(world, bp, vehicle_bp, spawn_points)
    vehicle.set_autopilot(False)
    current["vehicle"] = vehicle
    print(f"[EGO] Spawned: {vehicle.type_id}")

    #camera 
    cam_bp = bp.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(CAM_W))
    cam_bp.set_attribute("image_size_y", str(CAM_H))
    cam_bp.set_attribute("fov", str(FOV))
    cam_bp.set_attribute("sensor_tick", str(SENSOR_TICK))
    cam_transform = carla.Transform(carla.Location(x=1.5, z=2.2))

    latest_rgb = None
    camera_surface = None
    last_frame_time = [time.time()] 
    def on_image(image):
        nonlocal latest_rgb, camera_surface
        try:
            arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((CAM_H, CAM_W, 4))
            bgr = arr[:, :, :3]
            rgb = bgr[:, :, ::-1]
            latest_rgb = rgb
            camera_surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
            last_frame_time[0] = time.time()
        except Exception as e:
            print(f"[IMAGE_CB] error: {e}")

   #camera
    camera = create_camera(world, cam_bp, cam_transform, vehicle, on_image)
    current["camera"] = camera

    # third person following the car
    stop_spec = threading.Event()
    spec_thread = threading.Thread(
        target=follow_spectator,
        args=(lambda: current["vehicle"], lambda: current["world"], stop_spec),
        daemon=True
    )
    spec_thread.start()

    # Buffers
    images = []
    labels = []
    G_IMAGES = images
    G_LABELS = labels
    last_capture = None

    control = carla.VehicleControl()
    steer_cmd = 0.0
    STEER_DEADZONE = 0.08
    STEER_SMOOTH = 0.3
    TRIGGER_DZ = 0.05
    KEY_STEER_STEP = 0.05

    def respawn():
        nonlocal camera, world, bp, vehicle, latest_rgb, camera_surface, control, steer_cmd
        safe_stop_destroy_sensor(camera)
        camera = None
        current["camera"] = None

        # destroy tesla 
        try:
            vehicle.destroy()
        except:
            pass
        current["vehicle"] = None

        # respawn tesla 
        vehicle_new = try_spawn(world, bp, vehicle_bp, spawn_points)
        vehicle = vehicle_new
        current["vehicle"] = vehicle
        vehicle.set_autopilot(False)

     
        control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0)
        steer_cmd = 0.0
        latest_rgb = None
        camera_surface = None

        camera = create_camera(world, cam_bp, cam_transform, vehicle, on_image)
        current["camera"] = camera
        print("[EGO] Respawned.")

    def reconnect():
        nonlocal client, world, bp, vehicle, camera, spawn_points, spec_thread
        print("[RECONNECT] Attempting to reconnect to CARLA...")
        safe_stop_destroy_sensor(current.get("camera"))
        try:
            if current.get("vehicle"):
                current["vehicle"].destroy()
        except:
            pass
        current["camera"] = None
        current["vehicle"] = None

        ok = False
        for i in range(10):
            try:
                client, world, bp = connect_world()
                spawn_points = world.get_map().get_spawn_points()
                ok = True
                break
            except Exception as e:
                print(f"[RECONNECT] retry {i+1}/10 failed: {e}")
                time.sleep(1.0)
        if not ok:
            print("[RECONNECT] Failed to reconnect after retries.")
            return

        current["client"], current["world"], current["bp"] = client, world, bp

        try:
            vehicle = try_spawn(world, bp, vehicle_bp, spawn_points)
            vehicle.set_autopilot(False)
            current["vehicle"] = vehicle
            camera = create_camera(world, cam_bp, cam_transform, vehicle, on_image)
            current["camera"] = camera
            print("[RECONNECT] Reconnected and respawned.")
        except Exception as e:
            print(f"[RECONNECT] spawn failed: {e}")

        try:
            stop_spec.set()
            spec_thread.join(timeout=0.5)
        except:
            pass
        stop_spec.clear()
        spec_thread = threading.Thread(
            target=follow_spectator,
            args=(lambda: current["vehicle"], lambda: current["world"], stop_spec),
            daemon=True
        )
        spec_thread.start()

    #print instructions 
    print("Controls:")
    print("  Drive: Arrows (Left/Right steer, Up throttle, Space brake) | WASD also works for steer (A/D)")
    print("  Capture: G = GREEN light, R = RED light, U = undo last")
    print("  Misc: L = respawn, F5 = reconnect, ESC = quit")

    running = True
    can_respawn_key = True
    last_watchdog = time.time()

    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_F5:
                        reconnect()
                    elif event.key == pygame.K_l and can_respawn_key:
                        can_respawn_key = False
                        respawn()
                    elif event.key == pygame.K_g:
                        if latest_rgb is not None and len(labels) < TARGET_COUNT:
                            images.append(latest_rgb.copy())
                            labels.append(np.int8(1))
                            green_count = int(np.sum(np.array(labels, dtype=np.int8)))
                            red_count = len(labels) - green_count
                            print(f"[CAPTURE] GREEN  (Total: {len(labels)}/{TARGET_COUNT} | Green: {green_count} | Red: {red_count})")
                            last_capture = "GREEN"
                            if len(labels) % AUTOSAVE_EVERY == 0:
                                save_atomic(OUT_DIR, images, labels)
                            if len(labels) >= TARGET_COUNT:
                                running = False
                                print("[INFO] Target reached, stopping...")
                    elif event.key == pygame.K_r:
                        if latest_rgb is not None and len(labels) < TARGET_COUNT:
                            images.append(latest_rgb.copy())
                            labels.append(np.int8(0)) 
                            green_count = int(np.sum(np.array(labels, dtype=np.int8)))
                            red_count = len(labels) - green_count
                            print(f"[CAPTURE] RED    (Total: {len(labels)}/{TARGET_COUNT} | Green: {green_count} | Red: {red_count})")
                            last_capture = "RED"
                            if len(labels) % AUTOSAVE_EVERY == 0:
                                save_atomic(OUT_DIR, images, labels)
                            if len(labels) >= TARGET_COUNT:
                                running = False
                                print("[INFO] Target reached, stopping...")
                    elif event.key == pygame.K_u:
                        if images:
                            images.pop()
                            removed = labels.pop()
                            green_count = int(np.sum(np.array(labels, dtype=np.int8))) if labels else 0
                            red_count = len(labels) - green_count
                            last_capture = f"UNDO ({'GREEN' if removed==1 else 'RED'})"
                            print(f"[UNDO] Removed last capture. (Total: {len(labels)}/{TARGET_COUNT} | Green: {green_count} | Red: {red_count})")
                            save_atomic(OUT_DIR, images, labels)
                elif event.type == pygame.KEYUP:
                    if event.key == pygame.K_l:
                        can_respawn_key = True

            
            now = time.time()
            if now - last_watchdog > 1.0:
                last_watchdog = now
                try:
                    _ = world.get_snapshot()
                except Exception:
                    reconnect()
                else:
                  
                    if time.time() - last_frame_time[0] > 2.5:
                        print("[WATCHDOG] No frames in 2.5s, recreating camera...")
                        safe_stop_destroy_sensor(current.get("camera"))
                        camera = create_camera(world, cam_bp, cam_transform, vehicle, on_image)
                        current["camera"] = camera
                        last_frame_time[0] = time.time()

            # driving 
            keys = pygame.key.get_pressed()
            steer_target = 0.0
            throttle_target = 1.0 if keys[pygame.K_UP] else 0.0
            brake_target = 1.0 if keys[pygame.K_SPACE] else 0.0

            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                steer_target = clamp(steer_cmd - KEY_STEER_STEP, -1.0, 1.0)
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
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
                    throttle_target = 0.0 if r2 < TRIGGER_DZ else r2
                    brake_target = 0.0 if l2 < TRIGGER_DZ else l2
                except Exception:
                    pass

            steer_cmd = steer_cmd + (steer_target - steer_cmd) * STEER_SMOOTH
            control.steer = clamp(steer_cmd, -1.0, 1.0)
            control.throttle = clamp(throttle_target, 0.0, 1.0)
            control.brake = clamp(brake_target, 0.0, 1.0)
            control.reverse = False

            try:
                vehicle.apply_control(control)
            except Exception:
                pass

            if camera_surface is not None:
                screen.blit(camera_surface, (0, 0))
                font = pygame.font.SysFont(None, 18)
                total = len(labels)
                green_count = int(np.sum(np.array(labels, dtype=np.int8))) if labels else 0
                red_count = total - green_count
                hud1 = font.render(f"Steer {control.steer:+.2f}  Th {control.throttle:.2f}  Br {control.brake:.2f}", True, (255, 255, 255))
                hud2 = font.render(f"Total: {total}/{TARGET_COUNT}   Green: {green_count}   Red: {red_count}", True, (255, 255, 0))
                hud3 = font.render("G=Green  R=Red  U=Undo  L=Respawn  F5=Reconnect  ESC=Quit", True, (200, 255, 255))
                screen.blit(hud1, (6, 6))
                screen.blit(hud2, (6, 24))
                screen.blit(hud3, (6, 42))
                if last_capture:
                    hud4 = font.render(f"Last: {last_capture}", True, (255, 200, 0))
                    screen.blit(hud4, (6, 60))
                pygame.display.flip()

            clock.tick(FPS)

    except KeyboardInterrupt:
        pass
    finally:
       #save
        try:
            save_atomic(OUT_DIR, images, labels)
        except Exception as e:
            print(f"[SAVE] FAILED: {e}")

        try:
            safe_stop_destroy_sensor(current.get("camera"))
        except:
            pass
        try:
            if current.get("vehicle"):
                current["vehicle"].destroy()
        except:
            pass
        try:
            stop_spec.set()
        except:
            pass
        pygame.quit()
        print("[DONE]")

if __name__ == "__main__":
    main()
