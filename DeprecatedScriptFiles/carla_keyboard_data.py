import os
import re
import shutil
import carla
import pygame
import time
import threading
import math
import random
import numpy as np
from numpy.lib.format import open_memmap
from carla import VehicleLightState

WINDOW_WIDTH, WINDOW_HEIGHT = 320, 240
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
CHUNK_SIZE = 2000    
PRINT_EVERY = 200     

# Merge options
MERGE_IMAGES = True  
MERGE_LABELS = True   

HEADROOM_BYTES = 256 * 1024 * 1024 

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def speed_kmh(vehicle):
    v = vehicle.get_velocity()
    return 3.6 * math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z)

def disk_free_bytes(path):
    try:
        return shutil.disk_usage(path).free
    except Exception:
        # if the path doesn't exist at this moment
        base = path if os.path.isdir(path) else os.path.dirname(path) or "."
        return shutil.disk_usage(base).free

def atomic_save_npy(path, arr):
    tmp = path + ".tmp"
    np.save(tmp, arr)
    os.replace(tmp, path)

def atomic_save_npz(path, **arrays):
    tmp = path + ".tmp"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)

def follow_spectator(get_vehicle_fn, world, stop_event):
    spectator = world.get_spectator()
    back_distance = 6
    height = 3
    while not stop_event.is_set():
        v = get_vehicle_fn()
        if v is None:
            time.sleep(0.03); continue
        try:
            tf = v.get_transform()
        except RuntimeError:
            time.sleep(0.03); continue

        yaw_rad = math.radians(tf.rotation.yaw)
        offset_x = -back_distance * math.cos(yaw_rad)
        offset_y = -back_distance * math.sin(yaw_rad)
        cam_location = carla.Location(
            x=tf.location.x + offset_x,
            y=tf.location.y + offset_y,
            z=tf.location.z + height
        )
        cam_rotation = carla.Rotation(pitch=-10, yaw=tf.rotation.yaw)
        spectator.set_transform(carla.Transform(cam_location, cam_rotation))
        time.sleep(0.03)

def spawn_traffic(client, world, blueprint_library, ego_vehicle, num_vehicles=30):
    tm = client.get_trafficmanager(TRAFFIC_MANAGER_PORT)
    tm.set_global_distance_to_leading_vehicle(GLOBAL_LEAD_DIST)
    tm.set_synchronous_mode(False)
    tm.set_random_device_seed(SEED_TRAFFIC)

    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    ego_loc = ego_vehicle.get_transform().location
    filtered_points = [sp for sp in spawn_points if sp.location.distance(ego_loc) > 8.0]

    vehicle_bps = blueprint_library.filter('vehicle.*')
    batch, used = [], 0
    for sp in filtered_points:
        if used >= num_vehicles: break
        bp = random.choice(vehicle_bps)
        if bp.has_attribute('color'):
            bp.set_attribute('color', random.choice(bp.get_attribute('color').recommended_values))
        batch.append(carla.command.SpawnActor(bp, sp)); used += 1

    spawned_actors = []
    if batch:
        responses = client.apply_batch_sync(batch, True)
        actor_ids = [r.actor_id for r in responses if not r.error]
        if actor_ids:
            client.apply_batch_sync([carla.command.SetAutopilot(aid, True, tm.get_port()) for aid in actor_ids], True)
            for aid in actor_ids:
                a = world.get_actor(aid)
                if a: spawned_actors.append(a)
    print(f"[TRAFFIC] Spawned {len(spawned_actors)} NPC vehicles.")
    return spawned_actors

def relock_spectator_to(world, v):
    spectator = world.get_spectator()
    tf = v.get_transform()
    fwd = tf.get_forward_vector()
    cam_loc = tf.location - fwd * 6 + carla.Location(z=3)
    cam_rot = carla.Rotation(pitch=-10, yaw=tf.rotation.yaw)
    spectator.set_transform(carla.Transform(cam_loc, cam_rot))



def chunk_tag(i): return f"chunk_{i:05d}"

def save_chunk(out_dir, idx, images, angles, speeds, turn_signals, flag_left, flag_right, timestamps):
    """Write one chunk to disk. Clears buffers ONLY after a successful save."""
    if not images:
        return True

    tag = chunk_tag(idx)

    # save labels 
    try:
        atomic_save_npy(os.path.join(out_dir, f"{tag}_angles.npy"), np.asarray(angles, dtype=np.float32))
        atomic_save_npy(os.path.join(out_dir, f"{tag}_speeds_kmh.npy"), np.asarray(speeds, dtype=np.float32))
        atomic_save_npy(os.path.join(out_dir, f"{tag}_turn_signals.npy"), np.asarray(turn_signals, dtype=np.int8))
        atomic_save_npy(os.path.join(out_dir, f"{tag}_flag_left.npy"), np.asarray(flag_left, dtype=np.uint8))
        atomic_save_npy(os.path.join(out_dir, f"{tag}_flag_right.npy"), np.asarray(flag_right, dtype=np.uint8))
        atomic_save_npy(os.path.join(out_dir, f"{tag}_timestamps.npy"), np.asarray(timestamps, dtype=np.float64))
    except Exception as e:
        print(f"[SAVE][{tag}] FAILED to save labels: {e}")
        return False 

    # Save the images 
    H, W, C = images[0].shape
    n = len(images)
    needed = n * H * W * C
    free = disk_free_bytes(out_dir)

    try:
        if free >= needed + HEADROOM_BYTES:
            arr = np.stack(images).astype(np.uint8)
            atomic_save_npy(os.path.join(out_dir, f"{tag}_images.npy"), arr)
            print(f"[SAVE][{tag}] {n} frames -> NPY")
        else:
            arr = np.stack(images).astype(np.uint8)
            atomic_save_npz(os.path.join(out_dir, f"{tag}_images.npz"), imgs=arr)
            print(f"[SAVE][{tag}] {n} frames -> NPZ (compressed)")
    except Exception as e:
        print(f"[SAVE][{tag}] image save failed ({e}), trying smaller splits...")
        try:
            chunk = 200  
            for i in range(0, n, chunk):
                sub = np.stack(images[i:i+chunk]).astype(np.uint8)
                atomic_save_npz(os.path.join(out_dir, f"{tag}_images_{i//chunk:03d}.npz"), imgs=sub)
            print(f"[SAVE][{tag}] split NPZ saved in parts.")
        except Exception as e2:
            print(f"[SAVE][{tag}] FAILED to save images even in splits: {e2}")
            return False  

    # Everything saved properly 
    images.clear(); angles.clear(); speeds.clear()
    turn_signals.clear(); flag_left.clear(); flag_right.clear(); timestamps.clear()
    return True

def find_chunks(out_dir):

    bases = set()
    for f in os.listdir(out_dir):
        m = re.match(r'^(chunk_\d{5})_(images\.(npy|npz)|angles\.npy|speeds_kmh\.npy|turn_signals\.npy|flag_left\.npy|flag_right\.npy|timestamps\.npy)$', f)
        if m:
            bases.add(m.group(1))
    return sorted(bases)

def auto_merge(out_dir):
    bases = find_chunks(out_dir)
    if not bases:
        print("[MERGE] No chunks to merge.")
        return

    total = 0
    sizes = {}
    for b in bases:
        ang = np.load(os.path.join(out_dir, f"{b}_angles.npy"), mmap_mode='r')
        n = ang.shape[0]
        sizes[b] = n
        total += n
    print(f"[MERGE] Total frames: {total}")

    if MERGE_LABELS:
        angles = open_memmap(os.path.join(out_dir, "angles.npy"), mode="w+", dtype=np.float32, shape=(total,))
        speeds = open_memmap(os.path.join(out_dir, "speeds_kmh.npy"), mode="w+", dtype=np.float32, shape=(total,))
        signals = open_memmap(os.path.join(out_dir, "turn_signals.npy"), mode="w+", dtype=np.int8, shape=(total,))
        fleft  = open_memmap(os.path.join(out_dir, "flag_left.npy"), mode="w+", dtype=np.uint8, shape=(total,))
        fright = open_memmap(os.path.join(out_dir, "flag_right.npy"), mode="w+", dtype=np.uint8, shape=(total,))
        times  = open_memmap(os.path.join(out_dir, "timestamps.npy"), mode="w+", dtype=np.float64, shape=(total,))

        idx = 0
        for b in bases:
            ang = np.load(os.path.join(out_dir, f"{b}_angles.npy"), mmap_mode='r')
            spd = np.load(os.path.join(out_dir, f"{b}_speeds_kmh.npy"), mmap_mode='r')
            sig = np.load(os.path.join(out_dir, f"{b}_turn_signals.npy"), mmap_mode='r')
            fl  = np.load(os.path.join(out_dir, f"{b}_flag_left.npy"), mmap_mode='r')
            fr  = np.load(os.path.join(out_dir, f"{b}_flag_right.npy"), mmap_mode='r')
            ts  = np.load(os.path.join(out_dir, f"{b}_timestamps.npy"), mmap_mode='r')
            n = sizes[b]
            angles[idx:idx+n] = ang
            speeds[idx:idx+n] = spd
            signals[idx:idx+n] = sig
            fleft[idx:idx+n] = fl
            fright[idx:idx+n] = fr
            times[idx:idx+n] = ts
            idx += n
            print(f"[MERGE] Labels {b} -> {n} frames")

    if MERGE_IMAGES:
        # Get the shape from the first image
        first_path_npy = os.path.join(out_dir, f"{bases[0]}_images.npy")
        first_path_npz = os.path.join(out_dir, f"{bases[0]}_images.npz")
        if os.path.exists(first_path_npy):
            first_img = np.load(first_path_npy, mmap_mode='r')
            H, W, C = first_img.shape[1], first_img.shape[2], first_img.shape[3]
        else:
            first_npz = np.load(first_path_npz, mmap_mode='r')
            first_img = first_npz["imgs"]
            H, W, C = first_img.shape[1], first_img.shape[2], first_img.shape[3]
            first_npz.close()

        needed_bytes = total * H * W * C
        free = disk_free_bytes(out_dir)

        if needed_bytes + HEADROOM_BYTES > free:
            print(f"[MERGE] Not enough disk for images.npy (need ~{needed_bytes/1e9:.1f} GB, free ~{free/1e9:.1f} GB).")
            print("[MERGE] Keeping images chunked (npy/npz). Labels merged successfully.")
            return

        imgs = open_memmap(os.path.join(out_dir, "images.npy"), mode="w+", dtype=np.uint8, shape=(total, H, W, C))
        idx = 0
        for b in bases:
            npy_path = os.path.join(out_dir, f"{b}_images.npy")
            npz_path = os.path.join(out_dir, f"{b}_images.npz")
            if os.path.exists(npy_path):
                arr = np.load(npy_path, mmap_mode='r')
            else:
                npz = np.load(npz_path, mmap_mode='r')
                arr = npz["imgs"]
            n = arr.shape[0]
            imgs[idx:idx+n] = arr
            idx += n
            print(f"[MERGE] Images {b} -> {n} frames")
            
            try:
                npz.close()
            except:
                pass

    print("[MERGE] Done.")



def main():
    ensure_dir(OUT_DIR)
    print(f"[RUN] Saving to: {OUT_DIR}")

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("CARLA Manual Drive + Logger + Traffic (320x240)")
    clock = pygame.time.Clock()

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()

    #Spawns in tesla 
    vehicle_bp = bp_lib.find('vehicle.tesla.model3')
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        print("No spawn points found!"); return

    def try_spawn():
        random.shuffle(spawn_points)
        for sp in spawn_points:
            v = world.try_spawn_actor(vehicle_bp, sp)
            if v: return v
        return world.spawn_actor(vehicle_bp, random.choice(spawn_points))

    vehicle = try_spawn()
    print(f"[EGO] Spawned: {vehicle.type_id}")

    # This is the camera that follows around the car 
    camera_bp = bp_lib.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(CAM_W))
    camera_bp.set_attribute('image_size_y', str(CAM_H))
    camera_bp.set_attribute('fov', str(FOV))
    camera_bp.set_attribute('sensor_tick', str(SENSOR_TICK))
    cam_transform = carla.Transform(carla.Location(x=1.5, z=2.2))
    camera = world.spawn_actor(camera_bp, cam_transform, attach_to=vehicle)

    camera_np = None
    camera_surface = None
    def process_image(image):
        nonlocal camera_np, camera_surface
        arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((CAM_H, CAM_W, 4))
        rgb = arr[:, :, :3][:, :, ::-1]
        camera_np = rgb.copy()
        camera_surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
    camera.listen(process_image)

   
    current = {"vehicle": vehicle}
    stop_spec = threading.Event()
    spec_thread = threading.Thread(
        target=follow_spectator,
        args=(lambda: current["vehicle"], world, stop_spec),
        daemon=True
    )
    spec_thread.start()

    # Spawns in the other cars into the simulator 
    npc_vehicles = spawn_traffic(client, world, bp_lib, vehicle, NUM_NPC_VEHICLES)

    
    control = carla.VehicleControl()
    # turn signal that is -1 for left, 0 for none, and 1 for right
    turn_signal = 0 

    images, angles, speeds, turn_signals, flag_left, flag_right, timestamps = ([] for _ in range(7))
    chunk_idx = 0
    total_frames = 0
    can_respawn = True

    #respawn the car 
    def respawn_ego():
        nonlocal camera, vehicle, camera_np, camera_surface, control, turn_signal
        try:
            if camera.is_listening: camera.stop()
        except: pass
        try: camera.destroy()
        except: pass
        try: vehicle.destroy()
        except: pass

        v = try_spawn()
        vehicle = v
        current["vehicle"] = v
        vehicle.set_light_state(VehicleLightState.NONE)
        control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0)
        turn_signal = 0

        camera = world.spawn_actor(camera_bp, cam_transform, attach_to=vehicle)
        camera.listen(process_image)
        relock_spectator_to(world, vehicle)
        print("[EGO] Respawned.")

    try:
        print("Controls: Left/Right steer | Up throttle | Space brake | A/D left/right | R cancel | L respawn | ESC quit")
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
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

            # steering controls
            steer_step = 0.05
            if keys[pygame.K_LEFT]:
                control.steer = max(-1.0, control.steer - steer_step)
            elif keys[pygame.K_RIGHT]:
                control.steer = min(1.0, control.steer + steer_step)
            else:
                control.steer = 0.0

            # acceleration and brake 
            control.throttle = 1.0 if keys[pygame.K_UP] else 0.0
            control.brake = 1.0 if keys[pygame.K_SPACE] else 0.0
            control.reverse = False

        
            if keys[pygame.K_a]:
                vehicle.set_light_state(VehicleLightState.LeftBlinker); turn_signal = -1
            elif keys[pygame.K_d]:
                vehicle.set_light_state(VehicleLightState.RightBlinker); turn_signal = 1
            elif keys[pygame.K_r]:
                vehicle.set_light_state(VehicleLightState.NONE); turn_signal = 0
            else:
                ls = vehicle.get_light_state()
                turn_signal = -1 if ls == VehicleLightState.LeftBlinker else (1 if ls == VehicleLightState.RightBlinker else 0)

            vehicle.apply_control(control)

            
            if camera_surface is not None:
                screen.blit(camera_surface, (0, 0))
                font = pygame.font.SysFont(None, 18)
                spd = speed_kmh(vehicle)
                sig_str = "LEFT" if turn_signal == -1 else ("RIGHT" if turn_signal == 1 else "NONE")
                hud1 = font.render(f"Steer: {control.steer:+.2f}", True, (255, 255, 0))
                hud2 = font.render(f"Throttle:{control.throttle:.2f} Brake:{control.brake:.2f}", True, (200, 255, 255))
                hud3 = font.render(f"Speed: {spd:.1f} km/h  Signal: {sig_str}", True, (255, 200, 0))
                hud4 = font.render(f"Frames: {total_frames}", True, (255, 255, 255))
                screen.blit(hud1, (6, 6)); screen.blit(hud2, (6, 24))
                screen.blit(hud3, (6, 42)); screen.blit(hud4, (6, 60))
                pygame.display.flip()

           
            if camera_np is not None:
                images.append(camera_np)
                angles.append(np.float32(control.steer))
                speeds.append(np.float32(speed_kmh(vehicle)))
                turn_signals.append(np.int8(turn_signal))
                flag_left.append(np.uint8(turn_signal == -1))
                flag_right.append(np.uint8(turn_signal == 1))
                timestamps.append(np.float64(time.time()))
                total_frames += 1

                if total_frames % PRINT_EVERY == 0:
                    print(f"[LOG] frames={total_frames}")

                
                if len(images) >= CHUNK_SIZE:
                    chunk_idx += 1
                    ok = save_chunk(OUT_DIR, chunk_idx, images, angles, speeds, turn_signals, flag_left, flag_right, timestamps)
                    if not ok:
                        print("[SAVE] Chunk failed — data kept in RAM. Free up disk or change OUT_DIR and press ESC to exit safely.")
                        time.sleep(0.5)

            clock.tick(FPS)

    except KeyboardInterrupt:
        pass
    finally:
        print("[CLEANUP] Stopping and saving...")

        try:
            stop_spec.set()
            spec_thread.join(timeout=0.5)
        except: pass

        try:
            if camera.is_listening: camera.stop()
        except: pass

        # get rid of the traffic 
        for a in npc_vehicles:
            try:
                if a and a.is_alive: a.destroy()
            except: pass

        # destroy the tesla and the camera 
        try: camera.destroy()
        except: pass
        try:
            if vehicle and vehicle.is_alive: vehicle.destroy()
        except: pass

        pygame.quit()

        if images:
            chunk_idx += 1
            ok = save_chunk(OUT_DIR, chunk_idx, images, angles, speeds, turn_signals, flag_left, flag_right, timestamps)
            if not ok:
                print("[SAVE] Final chunk failed to save — out of space. Free space and rerun, or change OUT_DIR.")
                return

      
        auto_merge(OUT_DIR)
        print("[DONE] Final files saved in:", OUT_DIR)


if __name__ == "__main__":
    main()
