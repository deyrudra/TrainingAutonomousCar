import carla
import numpy as np
import cv2
import time
import threading
import random
from pathlib import Path
from carla import VehicleLightState


def collect_data(duration_sec, image_filename, angle_filename, signal_filename):
    Path("output").mkdir(parents=True, exist_ok=True)

    client = carla.Client('localhost', 2000)
    client.set_timeout(25.0)
    client.load_world("Town02")

    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.1  # 10 FPS
    world.apply_settings(settings)

    vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
    vehicle_bp.set_attribute('color', '255,0,255')
    
    # spawn_point = random.choice(world.get_map().get_spawn_points())
    # vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    spawn_points = world.get_map().get_spawn_points()
    vehicle = None
    for spawn_point in spawn_points:
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
        if vehicle is not None:
            break

    tm = client.get_trafficmanager()
    tm_port = tm.get_port()
    vehicle.set_autopilot(True, tm_port)
    tm.set_synchronous_mode(True)
    tm.ignore_lights_percentage(vehicle, 100)
    tm.update_vehicle_lights(vehicle, True)

    print(f"Spawned vehicle: {vehicle.type_id}")

    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '448')
    camera_bp.set_attribute('image_size_y', '252')
    camera_bp.set_attribute('fov', '145')
    camera_bp.set_attribute('sensor_tick', '0.1')
    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    image_list = []
    angle_list = []
    signal_list = []

    flipped_image_list = []
    flipped_angle_list = []
    flipped_signal_list = []

    latest_image = [None]

    def save_image(image):
        latest_image[0] = image

    camera.listen(save_image)

    spectator = world.get_spectator()

    def follow():
        while True:
            transform = vehicle.get_transform()
            spectator.set_transform(carla.Transform(
                transform.location + carla.Location(x=-6, z=3),
                transform.rotation))
            time.sleep(0.03)

    threading.Thread(target=follow, daemon=True).start()

    frame_count = int(duration_sec / settings.fixed_delta_seconds)
    for i in range(frame_count):
        world.tick()

        image = latest_image[0]
        if image is None:
            continue  # wait for first image

        #getting the vechile turn angle
        angle = np.clip(vehicle.get_control().steer, -1.0, 1.0)

        #getting the turn signal state
        light_state = vehicle.get_light_state()
        if light_state & VehicleLightState.LeftBlinker:
            turn_signal = -1
        elif light_state & VehicleLightState.RightBlinker:
            turn_signal = 1
        else:
            turn_signal = 0

        # Convert and store image
        array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        rgb = cv2.cvtColor(array, cv2.COLOR_BGRA2RGB)
        resized = cv2.resize(rgb, (244, 244))
        chw_image = np.transpose(resized, (2, 0, 1))

        image_list.append(resized)
        angle_list.append(angle)
        signal_list.append(turn_signal)


        # # === Augmented frame (flipped image, inverted angle/signal) ===
        # flipped_image = cv2.flip(resized, 1)  # Horizontal flip
        # flipped_angle = -angle
        # flipped_signal = -turn_signal if turn_signal != 0 else 0

        # flipped_image_list.append(flipped_image)
        # flipped_angle_list.append(flipped_angle)
        # flipped_signal_list.append(flipped_signal)


    # image_list.extend(flipped_image_list)
    # angle_list.extend(flipped_angle_list)
    # signal_list.extend(flipped_signal_list)

    # === Save results ===
    np.save(f"output/{image_filename}.npy", np.array(image_list))
    np.save(f"output/{angle_filename}.npy", np.array(angle_list))
    np.save(f"output/{signal_filename}.npy", np.array(signal_list))

    print(f"Saved {len(image_list)} images, angles, and signals.")

    # === Cleanup ===
    camera.stop()
    camera.destroy()
    vehicle.destroy()

    # Reset world settings
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)
    tm.set_synchronous_mode(False)
    print("Vehicle destroyed\n")


def main():
    collect_data(
        duration_sec=120,
        image_filename="small_images",
        angle_filename="small_angles",
        signal_filename="small_turn_signals"
    )

if __name__ == "__main__":
    main()
