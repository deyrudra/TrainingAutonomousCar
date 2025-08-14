import carla
import numpy as np
import cv2
import time
import threading
import random
from pathlib import Path


def collect_data(duration_sec, image_filename, velocity_filename):
    Path("output").mkdir(parents=True, exist_ok=True)

    client = carla.Client('localhost', 2000)
    client.set_timeout(25.0)
    client.load_world("Town02")

    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.1  # run at 10 FPS
    world.apply_settings(settings)

    #spawn the vec
    vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
    vehicle_bp.set_attribute('color', '255,0,255')

    spawn_points = world.get_map().get_spawn_points()
    vehicle = None
    for spawn_point in spawn_points:
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
        if vehicle is not None:
            break

    if vehicle is None:
        print("failed to spawn vehicle.")
        return

    #using traffic mamner to set up synchronous mode
    tm = client.get_trafficmanager()
    tm_port = tm.get_port()
    vehicle.set_autopilot(True, tm_port)
    tm.set_synchronous_mode(True)
    tm.update_vehicle_lights(vehicle, True)

    print(f"Spawned vehicle: {vehicle.type_id}")

    #spawn a handful of NPC vehicles with varied speeds
    background_vehicles = []
    for _ in range(10):
        npc_bp = random.choice(blueprint_library.filter('vehicle.*'))
        npc_spawn = random.choice(world.get_map().get_spawn_points())
        npc = world.try_spawn_actor(npc_bp, npc_spawn)
        if npc:
            npc.set_autopilot(True, tm_port)
            tm.vehicle_percentage_speed_difference(npc, random.randint(0, 30))
            background_vehicles.append(npc)
    print(f"Spawned {len(background_vehicles)} background vehicles.")

    #camera mounted on the ego vehicle
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '448')
    camera_bp.set_attribute('image_size_y', '252')
    camera_bp.set_attribute('fov', '145')
    camera_bp.set_attribute('sensor_tick', '0.1')
    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    image_list = []
    velocity_list = []
    latest_image = [None]

    def save_image(image):
        #keep the most recent frame for processing
        latest_image[0] = image

    camera.listen(save_image)

    #spectator camera follows behind the car
    spectator = world.get_spectator()
    def follow():
        while True:
            transform = vehicle.get_transform()
            spectator.set_transform(carla.Transform(
                transform.location + carla.Location(x=-6, z=3),
                transform.rotation))
            time.sleep(0.03)

    threading.Thread(target=follow, daemon=True).start()

    #collect frames and velocities for the requested duration
    frame_count = int(duration_sec / settings.fixed_delta_seconds)
    for i in range(frame_count):
        world.tick()

        image = latest_image[0]
        if image is None:
            continue  # wait until the first image arrives

        #velocity magnitude in m/s
        velocity_vector = vehicle.get_velocity()
        velocity = (velocity_vector.x**2 + velocity_vector.y**2 + velocity_vector.z**2)**0.5

        #convert CARLA BGRA to RGB, then resize to 244x244
        array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        rgb = cv2.cvtColor(array, cv2.COLOR_BGRA2RGB)
        resized = cv2.resize(rgb, (244, 244))

        image_list.append(resized)
        velocity_list.append(velocity)

    #write arrays to disk
    np.save(f"output/{image_filename}.npy", np.array(image_list))
    np.save(f"output/{velocity_filename}.npy", np.array(velocity_list))
    print(f"Saved {len(image_list)} images and velocities.")

    #tear down actors and restore world settings
    camera.stop()
    camera.destroy()
    vehicle.destroy()
    for npc in background_vehicles:
        npc.destroy()

    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)
    tm.set_synchronous_mode(False)

    print("Actors destroyed and world settings reset.\n")


def main():
    collect_data(
        duration_sec=400,
        image_filename="Town07_400s_images",
        velocity_filename="car_velocity"
    )

if __name__ == "__main__":
    main()
