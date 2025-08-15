# this script collects paired frames and labels from carla. we log images, steering angles, and turn signals.
# we run autopilot so data is steady. we keep timing in sync. we also spawn a few npcs for scene variety.
# the goal is a quick dataset writer that we can point at any town and let it run for a set duration.

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
    # we make sure the output folder exists. simple and safe.
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # we connect to the server and load the requested map. a bit of patience on big towns.
    client = carla.Client('localhost', 2000)
    client.set_timeout(25.0)
    client.load_world(map_name)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    # we switch to sync mode so sim time advances one tick per call. this keeps data aligned with control.
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.1
    world.apply_settings(settings)

    # we bring up the traffic manager in sync mode as well. this keeps autopilot deterministic-ish.
    tm = client.get_trafficmanager()
    tm_port = tm.get_port()
    tm.set_synchronous_mode(True)

    # we spawn our ego car. a tesla is fine. we add a bright color so it is easy to spot in the spectator cam.
    vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
    vehicle_bp.set_attribute('color', '255,0,255')
    spawn_point = random.choice(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    # we enable autopilot and set a few tm knobs. short gap. no slowdown. allow lane changes.
    # we also let it ignore lights to keep it moving for data. tune as needed for our task.
    vehicle.set_autopilot(True, tm_port)
    tm.vehicle_percentage_speed_difference(vehicle, 0)
    tm.auto_lane_change(vehicle, True)
    tm.ignore_lights_percentage(vehicle, 100)
    tm.distance_to_leading_vehicle(vehicle, 1.0)
    tm.update_vehicle_lights(vehicle, True)

    print(f"vehicle spawned at: {spawn_point.location}")

    # we add some background traffic for diversity. small number to avoid gridlock.
    background_vehicles = []
    for _ in range(10):
        npc_bp = random.choice(blueprint_library.filter('vehicle.*'))
        npc_spawn = random.choice(world.get_map().get_spawn_points())
        npc = world.try_spawn_actor(npc_bp, npc_spawn)
        if npc:
            npc.set_autopilot(True, tm_port)
            tm.vehicle_percentage_speed_difference(npc, random.randint(0, 30))
            background_vehicles.append(npc)

    # we attach a narrow fov camera on the hood. light and fast. ticks at 10 hz to match our step.
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '448')
    camera_bp.set_attribute('image_size_y', '252')
    camera_bp.set_attribute('fov', '145')
    camera_bp.set_attribute('sensor_tick', '0.1')
    cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(camera_bp, cam_transform, attach_to=vehicle)

    # we keep the latest frame in a tiny shared list. simple pattern. thread safe enough for this.
    latest_image = [None]
    def image_callback(img): latest_image[0] = img
    camera.listen(image_callback)

    # we set the spectator to follow a few meters back. this helps us watch behavior while logging.
    spectator = world.get_spectator()
    def follow():
        while True:
            tf = vehicle.get_transform()
            spectator.set_transform(carla.Transform(
                tf.location + carla.Location(x=-6, z=3),
                tf.rotation))
            time.sleep(0.03)

    threading.Thread(target=follow, daemon=True).start()

    # we collect three aligned streams: images, steering angle, and turn signal. all tick synced.
    images, angles, signals = [], [], []

    frame_count = int(duration_sec / settings.fixed_delta_seconds)
    for _ in range(frame_count):
        world.tick()
        img = latest_image[0]
        if img is None:
            continue

        # we read the live control from the ego. angle is already in [-1, 1]. we clamp just in case.
        steer = np.clip(vehicle.get_control().steer, -1.0, 1.0)
        light_state = vehicle.get_light_state()
        signal = -1 if light_state & VehicleLightState.LeftBlinker else 1 if light_state & VehicleLightState.RightBlinker else 0

        # we convert the raw bgra to rgb, then resize to 244x244. this size is enough for fast training.
        arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape((img.height, img.width, 4))
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
        resized = cv2.resize(rgb, (244, 244))

        images.append(resized)
        angles.append(steer)
        signals.append(signal)

    # we save each stream as a .npy. names are prefixed so we can shard runs. plz ensure disk has space.
    np.save(Path(output_dir) / f"{output_prefix}_images.npy", np.array(images))
    np.save(Path(output_dir) / f"{output_prefix}_angles.npy", np.array(angles))
    np.save(Path(output_dir) / f"{output_prefix}_turn_signals.npy", np.array(signals))
    print(f"saved {len(images)} frames to '{output_dir}' with prefix '{output_prefix}'.")

    # we tear everything down cleanly. vehicles first, then sensors. finally restore async mode.
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
    # we parse a few flags to keep usage simple. duration is in seconds. map is like town07. prefix tags the files.
    parser = argparse.ArgumentParser(description="collect carla data (images, steering angles, turn signals)")
    parser.add_argument("--duration", "-d", type=int, required=True, help="run duration in seconds")
    parser.add_argument("--map", "-m", dest="map_name", type=str, required=True, help="map name, e.g. Town07")
    parser.add_argument("--prefix", "-o", dest="output_prefix", type=str, required=True, help="output file prefix")
    parser.add_argument("--outdir", "-O", dest="output_dir", type=str, default="../output/partitions",
                        help="output directory (default: ../output/partitions)")
    args = parser.parse_args()

    collect_data(duration_sec=args.duration,
                 map_name=args.map_name,
                 output_dir=args.output_dir,
                 output_prefix=args.output_prefix)


if __name__ == "__main__":
    main()
