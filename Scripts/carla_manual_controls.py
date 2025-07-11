import carla
import pygame
import time
import random

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
vehicle_bp = blueprint_library.filter("vehicle.*model3*")[0]  # You can change the model
spawn_point = random.choice(world.get_map().get_spawn_points())
vehicle = world.spawn_actor(vehicle_bp, spawn_point)
vehicle.set_autopilot(False)

# === Spectator camera setup ===
spectator = world.get_spectator()

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

        # Apply control to vehicle
        vehicle.apply_control(control)

        # === Dynamic spectator follow camera ===
        transform = vehicle.get_transform()
        forward_vector = transform.get_forward_vector()
        cam_location = transform.location - forward_vector * 8 + carla.Location(z=3)
        cam_rotation = carla.Rotation(pitch=-10, yaw=transform.rotation.yaw)
        spectator.set_transform(carla.Transform(cam_location, cam_rotation))

        # Handle exit
        for event in pygame.event.get():
            if event.type == pygame.QUIT or keys[pygame.K_ESCAPE]:
                raise KeyboardInterrupt

except KeyboardInterrupt:
    print("Exiting and cleaning up...")

finally:
    vehicle.destroy()
    pygame.quit()
