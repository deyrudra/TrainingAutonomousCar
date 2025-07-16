import carla
import pygame
import time
import random
import numpy as np

# === Pygame setup ===
pygame.init()
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("CARLA Manual Control")

# === Connect to CARLA ===
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)
world = client.get_world()
blueprint_library = world.get_blueprint_library()

# === Spawn a vehicle ===
vehicle_bp = blueprint_library.filter("vehicle.*model3*")[0]
spawn_point = random.choice(world.get_map().get_spawn_points())
vehicle = world.spawn_actor(vehicle_bp, spawn_point)
vehicle.set_autopilot(False)

# === Spectator camera setup ===
spectator = world.get_spectator()

# === Attach a camera sensor ===
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '320')
camera_bp.set_attribute('image_size_y', '240')
camera_bp.set_attribute('fov', '110')  # Adjust FOV here (e.g. 90, 100, 110)
camera_bp.set_attribute('sensor_tick', '0.01')

camera_transform = carla.Transform(carla.Location(x=-7.5, z=6.4))  # hood-mounted camera
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)  


# Shared variable to store the latest camera frame
camera_image = None

def process_image(image):
    global camera_image
    # Convert raw data to numpy array (uint8)
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    # Reshape to (height, width, 4)
    array = array.reshape((image.height, image.width, 4))
    # Drop alpha channel and convert BGRA -> RGB by reversing last axis
    array = array[:, :, :3][:, :, ::-1]
    # Convert to surface for Pygame (swap axes to width x height)
    camera_image_raw = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    # Stretching image
    camera_image = pygame.transform.scale(camera_image_raw, (800, 600))
    
camera.listen(process_image)

# === Clock ===
clock = pygame.time.Clock()

print("Controls:")
print("Arrow keys: steer and drive")
print("Q/E: left/right turn signals")
print("ESC or close window: exit")

# === Main loop ===
try:
    while True:
        clock.tick(30)
        keys = pygame.key.get_pressed()
        control = carla.VehicleControl()

        # Forward/Reverse
        if keys[pygame.K_w]:
            control.throttle = 0.6
        elif keys[pygame.K_s]:
            control.brake = 0.5

        # Steering
        if keys[pygame.K_a]:
            control.steer = -0.5
        elif keys[pygame.K_d]:
            control.steer = 0.5

        # Turn signals
        control.left_indicator_light = keys[pygame.K_q]
        control.right_indicator_light = keys[pygame.K_e]

        vehicle.apply_control(control)

        # === Spectator follows vehicle ===
        transform = vehicle.get_transform()
        forward_vector = transform.get_forward_vector()
        cam_location = transform.location - forward_vector * 8 + carla.Location(z=3)
        cam_rotation = carla.Rotation(pitch=-10, yaw=transform.rotation.yaw)
        spectator.set_transform(carla.Transform(cam_location, cam_rotation))

        # === Display camera feed ===
        if camera_image:
            screen.blit(camera_image, (0, 0))
            pygame.display.flip()

        # Exit
        for event in pygame.event.get():
            if event.type == pygame.QUIT or keys[pygame.K_ESCAPE]:
                raise KeyboardInterrupt

except KeyboardInterrupt:
    print("Exiting and cleaning up...")

finally:
    camera.stop()
    vehicle.destroy()
    camera.destroy()
    pygame.quit()
