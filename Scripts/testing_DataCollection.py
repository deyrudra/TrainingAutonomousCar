import carla
import numpy as np
import cv2
import time
import threading
from pathlib import Path
from carla import VehicleLightState


def collect_data(duration_sec, image_filename, angle_filename, signal_filename):
    Path("output").mkdir(parents=True, exist_ok=True)

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    client.load_world("Town05")

    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    #for synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = True #set to true
    settings.fixed_delta_seconds = 0.1  # 10 FPS
    world.apply_settings(settings) #apply the settings, for suynch mode

    #spawn the car
    vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
    vehicle_bp.set_attribute('color', '255,255,0')
    spawn_point = world.get_map().get_spawn_points()[6]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)


    tm = client.get_trafficmanager()
    tm_port = tm.get_port()
    vehicle.set_autopilot(True, tm_port)#added with tm_port

    tm.set_synchronous_mode(True)
    tm.ignore_lights_percentage(vehicle, 100)
    tm.update_vehicle_lights(vehicle, True)

    print(f"Spawned vehicle: {vehicle.type_id}")

    #setting the camera for the data collection
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '320')
    camera_bp.set_attribute('image_size_y', '240')
    camera_bp.set_attribute('fov', '110')
    camera_bp.set_attribute('sensor_tick', '0.1')  # match fixed_delta_seconds
    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    #buffers that we add to
    image_list = []
    angle_list = []
    signal_list = []
    latest_image = [None]

    def save_image(image):
        latest_image[0] = image

    camera.listen(save_image)

    #follows the car with the spectator camera
    spectator = world.get_spectator()

    def follow():
        while True:
            transform = vehicle.get_transform()
            spectator.set_transform(carla.Transform(
                transform.location + carla.Location(x=-6, z=3),
                transform.rotation))
            time.sleep(0.03)

    threading.Thread(target=follow, daemon=True).start()

    #for actual simulation
    print("Starting data collection...")
    frame_count = int(duration_sec / settings.fixed_delta_seconds)
    for i in range(frame_count):
        world.tick()

        image = latest_image[0]
        if image is None:
            continue  # wait for first image

        # Vehicle state
        angle = np.clip(vehicle.get_control().steer, -1.0, 1.0)

        # # Set blinkers based on angle (optional)
        # if angle > 0.1:
        #     vehicle.set_light_state(VehicleLightState.RightBlinker)
        # elif angle < -0.1:
        #     vehicle.set_light_state(VehicleLightState.LeftBlinker)
        # else:
        #     vehicle.set_light_state(VehicleLightState.NONE)

        light_state = vehicle.get_light_state()
        if light_state & VehicleLightState.LeftBlinker:
            turn_signal = -1
        elif light_state & VehicleLightState.RightBlinker:
            turn_signal = 1
        else:
            turn_signal = 0

        # Convert and store image
        array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        # gray = cv2.cvtColor(array, cv2.COLOR_BGRA2GRAY)
        resized = cv2.resize(array, (160, 120))

        image_list.append(resized)
        angle_list.append(angle)
        signal_list.append(turn_signal)

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
        duration_sec=900,
        image_filename="extra_large_images",
        angle_filename="extra_large_angles",
        signal_filename="extra_large_turn_signals"
    )

if __name__ == "__main__":
    main()
