import carla
import numpy as np
import cv2
import time
import threading
import random
from pathlib import Path
from carla import VehicleLightState

#from agents.navigation.global_route_planner import GlobalRoutePlannerDAO, GlobalRoutePlanner


#car travels from point a-b, once we reach b, find a new b and travel there
def set_new_destination(vehicle, world):

    #get possible spawn points in the world along with current
    #locatoin and drive to a random spawn point
    spawn_points = world.get_map().get_spawn_points()
    current_location = vehicle.get_location()

    while True:
        dest = random.choice(spawn_points).location
        if dest.distance(current_location) > 35.0: #ensure the destination is far
            break

    vehicle.get_navigation().go_to_location(dest)
    return dest


def collect_data(duration_sec, image_filename, angle_filename, signal_filename):
    Path("output").mkdir(parents=True, exist_ok=True)

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    client.load_world("Town05")

    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.1  # 10 FPS
    world.apply_settings(settings)

    vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
    vehicle_bp.set_attribute('color', '255,255,0')
    spawn_point = random.choice(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

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

    print("Starting data collection...")
    start_time = time.time()
    current_dest = set_new_destination(vehicle, world)

    while time.time() - start_time < duration_sec:
        world.tick()

        image = latest_image[0]
        if image is None:
            continue

        #if the vechile is about to get to the destination set a new one
        if vehicle.get_location().distance(current_dest) < 5.0:
            current_dest = set_new_destination(vehicle, world) #change the destination

        #get the steering angle for the car at the frame
        angle = np.clip(vehicle.get_control().steer, -1.0, 1.0)

        #get the turn signal state
        light_state = vehicle.get_light_state()
        if light_state & VehicleLightState.LeftBlinker:
            turn_signal = -1
        elif light_state & VehicleLightState.RightBlinker:
            turn_signal = 1
        else:
            turn_signal = 0

        #saving 244x244 rgb images
        array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        rgb = cv2.cvtColor(array, cv2.COLOR_BGRA2RGB)
        resized = cv2.resize(rgb, (244, 244))
        chw_image = np.transpose(resized, (2, 0, 1)) 

        image_list.append(chw_image)
        angle_list.append(angle)
        signal_list.append(turn_signal)

    #save the collected data
    np.save(f"output/{image_filename}.npy", np.array(image_list))
    np.save(f"output/{angle_filename}.npy", np.array(angle_list))
    np.save(f"output/{signal_filename}.npy", np.array(signal_list))

    print(f"Saved {len(image_list)} images, angles, and signals.")

    #break the setup.
    camera.stop()
    camera.destroy()
    vehicle.destroy()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)
    tm.set_synchronous_mode(False)
    print("Vehicle destroyed\n")


def main():
    collect_data(
        duration_sec=900,
        image_filename="images",
        angle_filename="angles",
        signal_filename="turn_signals"
    )

if __name__ == "__main__":
    main()
