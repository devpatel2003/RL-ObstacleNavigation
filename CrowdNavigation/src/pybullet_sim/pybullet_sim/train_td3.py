import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.buffers import ReplayBuffer
from gymnasium import Wrapper

import torch.nn as nn
from td3_camera_gym import CrowdAvoidanceEnv
import time  # Needed for rendering
import pybullet as p
import shutil
import os
import torch
import numpy as np
import time


torch.backends.cudnn.benchmark = True  # Optimize CUDA performance

TIMESTEPS = 5_000_000 #500 ts = 1s

# Create training environment with GUI
def make_env(deterministic=False):
    def _init():
        env = CrowdAvoidanceEnv()
        env = Monitor(env)  
        return env
    return _init

# Enable TensorBoard logging

class ForwardBiasFirstActionWrapper(Wrapper):
    def __init__(self, env, forward_min=0.4):
        super().__init__(env)
        self.forward_min = forward_min
        self.first_step = True

    def reset(self, **kwargs):
        self.first_step = True
        return self.env.reset(**kwargs)

    def step(self, action):
        if self.first_step:
            action = np.array(action)
            action[0] = max(action[0], self.forward_min)
            self.first_step = False
        return self.env.step(action)

class DecayActionNoiseCallback(BaseCallback):
    def __init__(self, initial_sigma, final_sigma, decay_steps, verbose=0):
        super().__init__(verbose)
        self.initial_sigma = np.array(initial_sigma)
        self.final_sigma = np.array(final_sigma)
        self.decay_steps = decay_steps

    def _on_step(self) -> bool:
        progress = min(1.0, self.num_timesteps / self.decay_steps)
        new_sigma = (1 - progress) * self.initial_sigma + progress * self.final_sigma
        self.model.action_noise.sigma = new_sigma
        return True




if __name__ == "__main__":

    print("CUDA Available:", torch.cuda.is_available())
    print("Number of GPUs:", torch.cuda.device_count())
    print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")
    print("Torch Version:", torch.__version__)
    print("Torch CUDA Version:", torch.version.cuda)

    # Ensure fresh logs before training
    log_dir = "./td3_camera_logs/"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)  # Delete previous logs
    os.makedirs(log_dir, exist_ok=True)  # Create fresh log directory
    logger = configure(log_dir, ["stdout", "tensorboard"])

    n_envs = 10  
    env = SubprocVecEnv([make_env() for _ in range(n_envs)])

    # Set up evaluation callback
    eval_env = DummyVecEnv([make_env()])  
    eval_callback = EvalCallback(eval_env, best_model_save_path="./td3_camera_models/",
                                log_path=log_dir, eval_freq=50_000  , deterministic=True, render=False
        )

    # Define action noise 
    initial_sigma = [0.1, 0.1]  # high exploration
    final_sigma = [0.05, 0.02]  # stable fine-tuning
    action_noise = NormalActionNoise(mean=[0.1, 0], sigma=initial_sigma)

    decay_callback = DecayActionNoiseCallback(
    initial_sigma=initial_sigma,
    final_sigma=final_sigma,
    decay_steps=TIMESTEPS * 0.2  # How many steps it takes to reach final noise
)

    model = TD3(
        "MlpPolicy",
        env,
        learning_rate=1e-4,        # Learning rate
        batch_size=512,             # Batch size
        buffer_size=1_000_000,      # Experience replay buffer
        tau=0.005,                  # Target smoothing coefficient
        gamma=0.9999,                 # Future Discount factor
        train_freq=(4, "step"),  # Delay updates until new episode
        policy_delay=8,            # Delay policy updates
        gradient_steps=-1,         # Gradient updates per training iteration
        action_noise = action_noise,
        policy_kwargs={
            "net_arch": dict(pi=[800, 600], qf=[800, 600]),
            "activation_fn": torch.nn.ReLU,
        },
        verbose=1,  # Logging level (1 = info)
        tensorboard_log=log_dir,
        device="cuda"
    )

    # Load the old model weights into a new model with new LR
    old_model_path = "./td3_summer_model/best_model"  # Adjust path
    #model = TD3.load(old_model_path, env=env)
    #model.action_noise = action_noise  # Update action noise
    #model.load_replay_buffer("./td3_summer_models/replay_buffer_large.pkl")  # Load replay buffer

    '''model.replay_buffer = ReplayBuffer(
        buffer_size=5_000_000,                # ~25% of total steps
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=model.device,
        optimize_memory_usage=True
    )'''

    # Rebuild the model with new learning rates
    '''new_model = TD3(
        "MlpPolicy",
        env=env,
        learning_rate=1e-4,
        action_noise=model.action_noise,  # retain same noise
        policy_kwargs=model.policy_kwargs,
        buffer_size=model.replay_buffer.buffer_size,
        gamma=model.gamma,
        tau=model.tau,
        train_freq=model.train_freq,
        policy_delay=model.policy_delay,
        verbose=1,
        tensorboard_log=model.tensorboard_log,
        device=model.device
    )

    # Transfer weights
    new_model.policy.load_state_dict(model.policy.state_dict())
    print("✅ Loaded weights with new learning rate")'
    model = new_model
    '''

    model.set_logger(logger)

    #model = SAC.load("src/pybullet_sim/pybullet_sim/sac_models/sac_crowd_avoidance_1_50k", env=env)


    
    print("Model Created. Training begins...")


    try:
        model.learn(
            total_timesteps=TIMESTEPS,
            callback=eval_callback,
            progress_bar=True,
            reset_num_timesteps=True
        )
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted. Saving model...")
    finally:
        model.save("./td3_camera_models/td3_model_2")
        model.save_replay_buffer("./td3_camera_models/replay_buffer_2.pkl")
        print("[INFO] Model and replay buffer saved.")

    # Save final trained model


    #1 just target
    #2 lidar added
    #3 cnn added