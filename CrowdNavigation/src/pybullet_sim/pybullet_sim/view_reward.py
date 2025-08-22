import gymnasium as gym
import pybullet as p
import pybullet_data
from td3_camera_gym import CrowdAvoidanceEnv
import time
import keyboard
import matplotlib.pyplot as plt
import numpy as np



import torch 
import torch.nn as nn




# === Control Parameters ===
MAX_LINEAR_SPEED = 1
MAX_ANGULAR_SPEED = 1
show_lidar = False

env = CrowdAvoidanceEnv(use_gui=True)
obs = env.reset()

print("\n **Manual Drive Mode (WASD Controls)** ")
print("[W] = Forward | [S] = Backward | [A] = Left | [D] = Right | [ESC] = Exit")

def plot_lidar_scan(scan):
    plt.clf()
    angles = np.linspace(-np.pi, np.pi, len(scan))
    x = scan * np.cos(angles)
    y = scan * np.sin(angles)
    plt.scatter(x, y, s=2, color='red')
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.title("Real-time LiDAR Scan")
    plt.pause(0.00001)

if show_lidar:
    plt.ion()
    fig = plt.figure()

# === Main Loop ===
while True:
    

    linear_speed = 0.0
    angular_speed = 0.0

    if keyboard.is_pressed("up"):
        linear_speed = MAX_LINEAR_SPEED
    if keyboard.is_pressed("down"):
        linear_speed = -MAX_LINEAR_SPEED
    if keyboard.is_pressed("left"):
        angular_speed = MAX_ANGULAR_SPEED
    if keyboard.is_pressed("right"):
        angular_speed = -MAX_ANGULAR_SPEED
    if keyboard.is_pressed("f"):
        linear_speed = 10
    if keyboard.is_pressed("g"):
        angular_speed = 10
    if keyboard.is_pressed("esc"):
        print("\n Exiting manual control mode.")
        break

    action = [linear_speed, angular_speed]
    obs, reward, done, _, _ = env.step(action)

    lidar_scan = obs[7:79]

    if show_lidar:
        plot_lidar_scan(lidar_scan)



    print(f"Action: {action} | Env Reward: {reward:.3f} ")


    if done:
        print("\n Episode ended (Goal reached or collision). Restarting...")
        obs, _ = env.reset()

    time.sleep(0.0000001)

env.close()
