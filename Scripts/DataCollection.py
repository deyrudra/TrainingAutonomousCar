import carla
import numpy as np
import cv2
import time
import threading
import random
from pathlib import Path
from carla import VehicleLightState
import argparse


def collect_data(duration_sec: float,
                 map_name: str = "Town07",
                 output_dir: str = "../output/partitions",
                 output_prefix: str = "data") -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    client = carla.Client('localhost', 2000)
    client.set_timeout(25.0)
    client.load_world(map_name)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    # === Sync mode ===
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.1
    world.apply_settings(settings)

    # === Traffic Manager ===
    tm = client.get_trafficmanager()
    tm_port = tm.get_port()
    tm.set_synchronous_mode(True)

    # === Spawn ego vehicle ===
    vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
    vehicle_bp.set_attribute('color', '255,0,255')
    spawn_point = random.choice(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    # === Autopilot & Traffic Manager config ===
    vehicle.set_autopilot(True, tm_port)
    tm.vehicle_percentage_speed_difference(vehicle, 0)  # no slowdown
    tm.auto_lane_change(vehicle, True)
    tm.ignore_lights_percentage(vehicle, 100)
    tm.distance_to_leading_vehicle(vehicle, 1.0)
    tm.update_vehicle_lights(vehicle, True)

    print(f"Vehicle spawned at: {spawn_point.location}")

    # === Optional: spawn background traffic ===
    background_vehicles = []
    for _ in range(10):
        npc_bp = random.choice(blueprint_library.filter('vehicle.*'))
        npc_spawn = random.choice(world.get_map().get_spawn_points())
        npc = world.try_spawn_actor(npc_bp, npc_spawn)
        if npc:
            npc.set_autopilot(True, tm_port)
            tm.vehicle_percentage_speed_difference(npc, random.randint(0, 30))
            background_vehicles.append(npc)

    # === Setup camera ===
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '448')
    camera_bp.set_attribute('image_size_y', '252')
    camera_bp.set_attribute('fov', '145')
    camera_bp.set_attribute('sensor_tick', '0.1')
    cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(camera_bp, cam_transform, attach_to=vehicle)

    # === Camera capture ===
    latest_image = [None]
    def image_callback(img): latest_image[0] = img
    camera.listen(image_callback)

    # === Spectator follow ===
    spectator = world.get_spectator()
    def follow():
        while True:
            tf = vehicle.get_transform()
            spectator.set_transform(carla.Transform(
                tf.location + carla.Location(x=-6, z=3),
                tf.rotation))
            time.sleep(0.03)

    threading.Thread(target=follow, daemon=True).start()

    # === Data Buffers ===
    images, angles, signals = [], [], []

    frame_count = int(duration_sec / settings.fixed_delta_seconds)
    for _ in range(frame_count):
        world.tick()
        img = latest_image[0]
        if img is None:
            continue

        # === Extract controls ===
        steer = np.clip(vehicle.get_control().steer, -1.0, 1.0)
        light_state = vehicle.get_light_state()
        signal = -1 if light_state & VehicleLightState.LeftBlinker else 1 if light_state & VehicleLightState.RightBlinker else 0

        # === Image conversion ===
        arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape((img.height, img.width, 4))
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
        resized = cv2.resize(rgb, (244, 244))

        images.append(resized)
        angles.append(steer)
        signals.append(signal)

    # === Save data ===
    np.save(Path(output_dir) / f"{output_prefix}_images.npy", np.array(images))
    np.save(Path(output_dir) / f"{output_prefix}_angles.npy", np.array(angles))
    np.save(Path(output_dir) / f"{output_prefix}_turn_signals.npy", np.array(signals))
    print(f"Saved {len(images)} frames to '{output_dir}' with prefix '{output_prefix}'.")

    # === Cleanup ===
    camera.stop()
    camera.destroy()
    vehicle.destroy()
    for npc in background_vehicles:
        try:
            npc.destroy()
        except:
            pass
    world.apply_settings(carla.WorldSettings(synchronous_mode=False))
    tm.set_synchronous_mode(False)


def main():
    parser = argparse.ArgumentParser(description="Collect CARLA data")
    parser.add_argument("--duration", "-d", type=int, required=True, help="Run duration (seconds)")
    parser.add_argument("--map", "-m", dest="map_name", type=str, required=True, help="Map name (e.g. Town07)")
    parser.add_argument("--prefix", "-o", dest="output_prefix", type=str, required=True, help="Output file prefix")
    args = parser.parse_args()

    collect_data(duration_sec=args.duration, map_name=args.map_name, output_prefix=args.output_prefix)


if __name__ == "__main__":
    main()
