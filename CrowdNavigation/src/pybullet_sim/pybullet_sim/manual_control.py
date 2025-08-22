import gymnasium as gym
import pybullet as p
import pybullet_data
from rppo_gym import CrowdAvoidanceEnv
import time
import keyboard  # ✅ Used for key detection
import matplotlib.pyplot as plt
import numpy as np
import math

# Define movement speeds
MAX_LINEAR_SPEED = 1  # m/s
MAX_ANGULAR_SPEED = 1  # rad/s
show_lidar = True
show_speed = True
robot_state = 5
lidar_range = 1.0  # Maximum range of LiDAR in meters
lidar_size = 60
env = CrowdAvoidanceEnv(use_gui=True)
obs, _ = env.reset()

# --- speed-logging buffers ------------------------------------------
log_cmd = []   # desired linear speed  (m/s)
log_act = []   # executed linear speed (m/s)
log_t   = []   # timestep index        (env step number)


print("\n **Manual Drive Mode (WASD Controls)** ")
print("[W] = Forward | [S] = Backward | [A] = Left | [D] = Right | [ESC] = Exit")

def plot_lidar_scan(scan):
    plt.clf()                      # Clear previous frame
    angles = np.linspace(
        -math.pi / 2,           # –90 °
        math.pi / 2,             # +90 °
        len(scan),
        endpoint=False
    )
    x = -scan * np.sin(angles)      # Polar → Cartesian
    y = scan * np.cos(angles)

    plt.scatter(x, y, s=2, color="red")

    # Front half-plane only: x is now ≥ 0
    plt.ylim(0, lidar_range)       # forward direction
    plt.xlim(-lidar_range, lidar_range)
    plt.gca().set_aspect("equal", adjustable="box")

    plt.xlabel("X (m)  – left ↔ right")
    plt.ylabel("Y (m)  – forward")
    plt.title("Real-time LiDAR (front 180 °)")
    plt.pause(0.001)               # Tiny pause to refresh
if show_lidar:
    plt.ion()  # Interactive plotting mode
    fig = plt.figure()  # Create figure for LiDAR plot



# Loop indefinitely (exit when `ESC` is pressed)
while True:
    #env.render()  # Ensure PyBullet GUI is active

    

    # Default no movement
    linear_speed = 0.0
    angular_speed = 0.0

    # Read keyboard inputs
    if keyboard.is_pressed("up"):  # Forward
        linear_speed = MAX_LINEAR_SPEED
    if keyboard.is_pressed("down"):  # Backward
        linear_speed = -MAX_LINEAR_SPEED
    if keyboard.is_pressed("left"):  # Turn Left
        angular_speed = MAX_ANGULAR_SPEED
    if keyboard.is_pressed("right"):  # Turn Right
        angular_speed = -MAX_ANGULAR_SPEED
    if keyboard.is_pressed("i"): 
        linear_speed = 10
    if keyboard.is_pressed("l"):  # Turn Left
        angular_speed = 10
    if keyboard.is_pressed("j"):  # Turn Right
        angular_speed = -10

    # Exit simulation if `ESC` is pressed
    if keyboard.is_pressed("esc"):
        print("\n Exiting manual control mode.")
        break

    # Send action to environment
    action = [linear_speed, angular_speed]  # [v, w] format
    obs, reward, done, _, info = env.step(action)
    lidar_scan = obs[-lidar_size:]  # Extract normalized LiDAR scan from observation
    #lidar_scan = env.raw_lidar_scan  # Extract LiDAR scan from observation

    # appended to buffers
    log_cmd.append(linear_speed)
    log_act.append(env.act_lin_speed)     
    log_t.append(env.current_step)        

    if show_lidar:
        plot_lidar_scan(lidar_scan)  # Update plot



    # Print debug info (optional)
    #print(f"Action: {action} | Reward: {reward:.3f} | min: {min(env.raw_lidar_scan)}")



    # Reset environment if done
    if done:
        print("\n Episode ended (Goal reached or collision). Restarting...")
        print(info)
        obs = env.reset()

    time.sleep(0.0000001)  # Control refresh rate (adjust for smoothness)

# Close the environment
if log_cmd and show_speed:                             # non-empty?
    # Convert step index to seconds   (optional)
    CTRL_DT = 1/25                      # <- whatever you use in env
    time_axis = [k * CTRL_DT for k in log_t]

    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, log_cmd, label="Desired $v$")
    plt.plot(time_axis, log_act, label="Actual $v$")
    plt.xlabel("Time (s)")
    plt.ylabel("Linear speed (m/s)")
    plt.title("Desired vs. Executed Forward Speed")
    plt.legend()
    plt.tight_layout()
plt.show()
env.close()