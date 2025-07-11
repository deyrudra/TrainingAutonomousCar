import carla
import numpy as np
import cv2
import time
import threading
from pathlib import Path

def collect_data(duration_sec, image_filename, angle_filename):
    # Ensure output directory exists
    Path("output").mkdir(parents=True, exist_ok=True)

    # Connect to CARLA
    client = carla.Client('192.168.86.116', 2000)
    client.set_timeout(10.0)

    client.load_world("Town05")

    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    #spawn tesla car
    vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
    vehicle_bp.set_attribute('color', '255,255,0')
    spawn_point = world.get_map().get_spawn_points()[6]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    vehicle.set_autopilot(True)

    tm = client.get_trafficmanager()
    tm.ignore_lights_percentage(vehicle, 100)

    print(f"Spawned vehicle: {vehicle.type_id}")

    # Configure RGB camera
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '320')
    camera_bp.set_attribute('image_size_y', '240')
    camera_bp.set_attribute('fov', '110')
    camera_bp.set_attribute('sensor_tick', '0.1')  # 10 FPS
    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    # Data buffers
    image_list = []
    angle_list = []

    def save_image(image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        gray = cv2.cvtColor(array, cv2.COLOR_BGRA2GRAY)
        resized = cv2.resize(gray, (80, 60))
        angle = np.clip(vehicle.get_control().steer, -1.0, 1.0)
        image_list.append(resized)
        angle_list.append(angle)

    camera.listen(save_image)

    # # Optional: Spectator follows vehicle
    # spectator = world.get_spectator()
    # def follow():
    #     while True:
    #         transform = vehicle.get_transform()
    #         spectator.set_transform(carla.Transform(
    #             transform.location + carla.Location(x=-6, z=3),
    #             transform.rotation))
    #         time.sleep(0.03)
    # threading.Thread(target=follow, daemon=True).start()

    # Run for specified duration
    time.sleep(duration_sec)

    # Stop and save
    camera.stop()
    np.save(f"output/{image_filename}.npy", np.array(image_list))
    np.save(f"output/{angle_filename}.npy", np.array(angle_list))
    print(f"Saved {len(image_list)} images and angles to 'output/{image_filename}.npy' and '{angle_filename}.npy'")

    # Cleanup
    camera.destroy()
    vehicle.destroy()
    print("Vehicle destroyed\n")

def main():
    collect_data(duration_sec=60,  image_filename="small_lowres_images",  angle_filename="small_lowres_angles")
    time.sleep(5)
    collect_data(duration_sec=180, image_filename="medium_lowres_images", angle_filename="medium_lowres_angles")
    time.sleep(5)
    collect_data(duration_sec=300, image_filename="large_lowres_images",  angle_filename="large_lowres_angles")

if __name__ == "__main__":
    main()
