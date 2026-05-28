import gymnasium as gym
import pybullet as p
from stable_baselines3 import PPO, TD3
from td3_camera_gym import CrowdAvoidanceEnv
import time
import matplotlib.pyplot as plt
import numpy as np

# Load trained model
model = TD3.load("./td3_camera_models/td3_model_2")  # Adjust path to your model
#model = PPO.load("./big_models/full_model_1")
#model = PPO.load("./ppo_models/ppo_goal_nav")

# Create environment with GUI
env = CrowdAvoidanceEnv(use_gui=True)

# Reset environment
obs, _ = env.reset(seed=9)  # Set seed for reproducibility

def plot_lidar_scan(scan):
    plt.clf()  # Clear previous frame
    angles = np.linspace(-np.pi, np.pi, len(scan))  # Angles from -pi to pi
    x = scan * np.cos(angles)  # Convert polar to cartesian
    y = scan * np.sin(angles)
    plt.scatter(x, y, s=2, color='red')  # Plot scan points
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.title("Real-time LiDAR Scan")
    plt.pause(0.01)  # Small pause to update plot



# Run for 1000 steps (or until the episode ends)
i = 0
j = 0
for _ in range(200_000_000):

    env.render()  # Ensure PyBullet GUI is active
    #action, _ = model.predict(obs)  # Get action from trained model
    #obs, reward, done, _ = env.step(action)
    #lidar_scan = obs[7:457]

    # Get action from model
    action, _ = model.predict(obs, deterministic=True)

    # Step the environment
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    lidar_scan = obs[4:]
    #plot_lidar_scan(lidar_scan)  # Plot LiDAR scan
    
    i += 1
    j+= 1
    print(f"Reward: {reward:.3f}, Min: {min(lidar_scan)}, Step:{i} ")  # ? Print reward to track behavior
    #print(f"Action: {action}")  # ? Print action to track behavior
    if done:  
        i = 0
        print("Restarting environment.")
        obs, _ = env.reset(seed=j)
    

    #time.sleep(0.05)  # ? Slow down simulation for visibility


#  Keep the environment open
print("?? Testing Complete! Close window manually.")
while True:
    time.sleep(1)

