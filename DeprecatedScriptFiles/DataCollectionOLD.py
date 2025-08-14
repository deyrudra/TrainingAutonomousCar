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
                 map_name: str = "Town?",
                 output_dir: str = "../output/partitions",
                 output_prefix: str = "data") -> None:
    """
    Collects camera images, steering angles, and turn signals from a CARLA simulation.

    Args:
        duration_sec (float): Total duration to run the simulation (seconds).
        map_name (str): Name of the CARLA map to load (e.g., "Town07").
        output_dir (str): Directory where output .npy files will be saved.
        output_prefix (str): Filename prefix for saved .npy arrays.
    """
    # Prepare output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Connect to CARLA and load the specified map
    client = carla.Client('localhost', 2000)
    client.set_timeout(25.0)
    client.load_world(map_name)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    # Enable synchronous mode for consistent frame timing
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.1  # 10 FPS
    world.apply_settings(settings)

    # Spawn ego vehicle
    vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
    vehicle_bp.set_attribute('color', '255,0,255')
    # spawn_points = world.get_map().get_spawn_points()
    # vehicle = None
    # for sp in spawn_points:
    #     vehicle = world.try_spawn_actor(vehicle_bp, sp)
    #     if vehicle:
    #         break
    # if vehicle is None:
    #     raise RuntimeError("Failed to spawn vehicle on any spawn point.")

    spawn_point = random.choice(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    # Configure traffic manager to ignore lights
    tm = client.get_trafficmanager()
    tm_port = tm.get_port()
    vehicle.set_autopilot(True, tm_port)
    tm.set_synchronous_mode(True)
    tm.ignore_lights_percentage(vehicle, 100)

    # Set up camera sensor attached to vehicle
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '448')
    camera_bp.set_attribute('image_size_y', '252')
    camera_bp.set_attribute('fov', '145')
    camera_bp.set_attribute('sensor_tick', '0.1')
    cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(camera_bp, cam_transform, attach_to=vehicle)

    # Buffers for asynchronously received images
    latest_image = [None]
    def image_callback(img):
        latest_image[0] = img
    camera.listen(image_callback)

    # Spawn a spectator to follow the vehicle
    spectator = world.get_spectator()
    def follow_vehicle():
        while True:
            tf = vehicle.get_transform()
            spectator.set_transform(
                carla.Transform(
                    tf.location + carla.Location(x=-6, z=3),
                    tf.rotation
                )
            )
            time.sleep(0.03)

    threading.Thread(target=follow_vehicle, daemon=True).start()

    # Prepare data lists
    images, angles, signals = [], [], []

    # Run simulation loop
    frame_count = int(duration_sec / settings.fixed_delta_seconds)
    for _ in range(frame_count):
        world.tick()
        img = latest_image[0]
        if img is None:
            continue  # skip until first image arrives

        # Extract steering angle
        steer = np.clip(vehicle.get_control().steer, -1.0, 1.0)
        # Determine turn signal state
        ls = vehicle.get_light_state()
        if ls & VehicleLightState.LeftBlinker:
            sig = -1
        elif ls & VehicleLightState.RightBlinker:
            sig = 1
        else:
            sig = 0

        # Convert raw image to RGB, resize, and transpose
        arr = np.frombuffer(img.raw_data, dtype=np.uint8)
        arr = arr.reshape((img.height, img.width, 4))
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
        resized = cv2.resize(rgb, (244, 244))
        images.append(resized)
        angles.append(steer)
        signals.append(sig)

    # Save collected data
    np.save(Path(output_dir) / f"{output_prefix}_images.npy", np.array(images))
    np.save(Path(output_dir) / f"{output_prefix}_angles.npy", np.array(angles))
    np.save(Path(output_dir) / f"{output_prefix}_turn_signals.npy", np.array(signals))
    print(f"Saved {len(images)} frames to '{output_dir}' with prefix '{output_prefix}'.")

    # Cleanup
    camera.stop()
    camera.destroy()
    vehicle.destroy()
    world.apply_settings(carla.WorldSettings(synchronous_mode=False))
    tm.set_synchronous_mode(False)

def main():
    parser = argparse.ArgumentParser(
        description="Collect CARLA data for a single map run"
    )
    parser.add_argument(
        "--duration", "-d",
        type=int,
        required=True,
        help="Duration of the run in seconds"
    )
    parser.add_argument(
        "--map", "-m",
        dest="map_name",
        type=str,
        required=True,
        help="CARLA town name (e.g. Town07)"
    )
    parser.add_argument(
        "--prefix", "-o",
        dest="output_prefix",
        type=str,
        required=True,
        help="Output file prefix"
    )
    args = parser.parse_args()

    collect_data(
        duration_sec=args.duration,
        map_name=args.map_name,
        output_prefix=args.output_prefix
    )

if __name__ == "__main__":
    main()


# if __name__ == "__main__":

#     # Part 1
#     collect_data(duration_sec=80, map_name="Town07", output_prefix="Town07_80s")
#     collect_data(duration_sec=80, map_name="Town06", output_prefix="Town06_80s")
#     collect_data(duration_sec=80, map_name="Town05", output_prefix="Town05_80s")
#     collect_data(duration_sec=80, map_name="Town04", output_prefix="Town04_80s")
#     collect_data(duration_sec=80, map_name="Town03", output_prefix="Town03_80s")
#     collect_data(duration_sec=80, map_name="Town02", output_prefix="Town02_80s")
#     collect_data(duration_sec=80, map_name="Town01", output_prefix="Town01_80s")

#     # Part 2
#     collect_data(duration_sec=80, map_name="Town07", output_prefix="Town07_80s-2")
#     collect_data(duration_sec=80, map_name="Town06", output_prefix="Town06_80s-2")
#     collect_data(duration_sec=80, map_name="Town05", output_prefix="Town05_80s-2")
#     collect_data(duration_sec=80, map_name="Town04", output_prefix="Town04_80s-2")
#     collect_data(duration_sec=80, map_name="Town03", output_prefix="Town03_80s-2")
#     collect_data(duration_sec=80, map_name="Town02", output_prefix="Town02_80s-2")
#     collect_data(duration_sec=80, map_name="Town01", output_prefix="Town01_80s-2")

#     # Part 3
#     collect_data(duration_sec=80, map_name="Town07", output_prefix="Town07_80s-3")
#     collect_data(duration_sec=80, map_name="Town06", output_prefix="Town06_80s-3")
#     collect_data(duration_sec=80, map_name="Town05", output_prefix="Town05_80s-3")
#     collect_data(duration_sec=80, map_name="Town04", output_prefix="Town04_80s-3")
#     collect_data(duration_sec=80, map_name="Town03", output_prefix="Town03_80s-3")
#     collect_data(duration_sec=80, map_name="Town02", output_prefix="Town02_80s-3")
#     collect_data(duration_sec=80, map_name="Town01", output_prefix="Town01_80s-3")

#     # Part 4
#     collect_data(duration_sec=80, map_name="Town07", output_prefix="Town07_80s-4")
#     collect_data(duration_sec=80, map_name="Town06", output_prefix="Town06_80s-4")
#     collect_data(duration_sec=80, map_name="Town05", output_prefix="Town05_80s-4")
#     collect_data(duration_sec=80, map_name="Town04", output_prefix="Town04_80s-4")
#     collect_data(duration_sec=80, map_name="Town03", output_prefix="Town03_80s-4")
#     collect_data(duration_sec=80, map_name="Town02", output_prefix="Town02_80s-4")
#     collect_data(duration_sec=80, map_name="Town01", output_prefix="Town01_80s-4")
