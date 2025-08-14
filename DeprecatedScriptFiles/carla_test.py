import carla
import random
import time

def main():
    # Connecting to the CARLA Simulator
    client = carla.Client('174.138.208.33', 2000)
    client.set_timeout(10.0)

    # Getting CARLA world
    world = client.get_world()

    # Getting the collection of possible objects in the world (blueprint)
    blueprint_library = world.get_blueprint_library()

    # # Getting All Possible Vehicles (Filtering for Vehicles out of objects)
    # vehicle_blueprints = blueprint_library.filter('vehicle.*')

    # # Choosing a random vehicle blueprint
    # vehicle_bp = random.choice(vehicle_blueprints)

    # Choosing a Red Tesla Model 3
    vehicle_bp = blueprint_library.find('vehicle.tesla.model3')

    # Get a random spawn point
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = spawn_points[6]

    # Spawn the vehicle
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    vehicle.set_autopilot(True)
    print(f"Spawned vehicle: {vehicle.type_id}")


    # --- Create RGB Camera Sensor ---
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '320')
    camera_bp.set_attribute('image_size_y', '240')
    camera_bp.set_attribute('fov', '110')  # Adjust FOV here (e.g. 90, 100, 110)
    camera_bp.set_attribute('sensor_tick', '0.05') # 1 Frame every .05 seconds
    # camera_bp.set_attribute('color_converter', 'Raw')

    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))  # hood-mounted camera
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)  

    def save_image(image):
        image.convert(carla.ColorConverter.Raw)  # Convert before saving
        image.save_to_disk('output/fixed_%06d.png' % image.frame)

    camera.listen(save_image)

    # camera.listen(lambda image: image.save_to_disk('output/%d064.png'%image.frame))

    # Optional: make spectator follow car
    spectator = world.get_spectator()
    def follow():
        while True:
            transform = vehicle.get_transform()
            spectator.set_transform(carla.Transform(
                transform.location + carla.Location(x=-6, z=3),
                transform.rotation))
            time.sleep(0.03)

    import threading
    threading.Thread(target=follow, daemon=True).start()

    # Let it sit in the world for a bit
    time.sleep(30)

    # === Clean up ===
    camera.stop()
    camera.destroy()

    # Clean up (destroy the vehicle)
    vehicle.destroy()
    print("Vehicle destroyed")

if __name__ == '__main__':
    main()