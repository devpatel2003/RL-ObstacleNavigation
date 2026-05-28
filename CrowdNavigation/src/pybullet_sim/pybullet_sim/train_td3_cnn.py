import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.buffers import ReplayBuffer

from gymnasium import Wrapper

import torch.nn as nn
from td3_gym import CrowdAvoidanceEnv
import pybullet as p
import shutil
import os
import torch
import numpy as np
import time

torch.backends.cudnn.benchmark = True  # Optimize CUDA performance

TIMESTEPS = 30_000_000 #500 ts = 1s

# Create training environment with GUI
def make_env(rank, base_seed=42):
    def _init():
        seed = int((time.time() * 1e6) % 1e9) + rank * 1000 + np.random.randint(0, 1000)
        env = CrowdAvoidanceEnv()
        env = Monitor(env)
        env.reset(seed=seed) 
        return env
    return _init

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

class LiDARFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)

        self.raw_obs_dim = observation_space.shape[0]
        self.lidar_dim = 64
        self.meta_dim = self.raw_obs_dim - self.lidar_dim

        self.lidar_cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy_lidar = torch.zeros(1, 1, self.lidar_dim)
            cnn_output_dim = self.lidar_cnn(dummy_lidar).shape[1]

        self._features_dim = self.meta_dim + cnn_output_dim

    def forward(self, observations):
        meta = observations[:, :self.meta_dim]
        lidar = observations[:, self.meta_dim:].unsqueeze(1)
        lidar_feat = self.lidar_cnn(lidar)
        return torch.cat([meta, lidar_feat], dim=1)

if __name__ == "__main__":

    print("CUDA Available:", torch.cuda.is_available())
    print("Number of GPUs:", torch.cuda.device_count())
    print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")
    print("Torch Version:", torch.__version__)
    print("Torch CUDA Version:", torch.version.cuda)

    # Ensure fresh logs before training
    log_dir = "./td3_summer_logs/"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)  # Delete previous logs
    os.makedirs(log_dir, exist_ok=True)  # Create fresh log directory
    logger = configure(log_dir, ["stdout", "tensorboard"])

    n_envs = 16 
    env = SubprocVecEnv([make_env(rank=i) for i in range(n_envs)])

    # Set up evaluation callback
    eval_env = DummyVecEnv([make_env(10)])  
    eval_callback = EvalCallback(eval_env, best_model_save_path="./td3_summer_models/",
                                log_path=log_dir, eval_freq=500_000, deterministic=True, render=False)

    # Define action noise 
    initial_sigma = [0.1, 0.1]  # high exploration
    final_sigma = [0.05, 0.02]  # stable fine-tuning
    action_noise = NormalActionNoise(mean=[0, 0], sigma=initial_sigma)

    decay_callback = DecayActionNoiseCallback(
        initial_sigma=initial_sigma,
        final_sigma=final_sigma,
        decay_steps=TIMESTEPS * 0.25
    )

    

    '''model = TD3(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        batch_size=128,
        buffer_size=1_000_000,
        tau=0.005,
        gamma=0.995,
        train_freq=(1, "step"),
        policy_delay=2,
        gradient_steps=-1,
        action_noise=action_noise,
        policy_kwargs={
            "features_extractor_class": LiDARFeatureExtractor,
            "features_extractor_kwargs": dict(features_dim=128),
            "net_arch": dict(pi=[800, 600], qf=[800, 600]),
            "activation_fn": torch.nn.ReLU,
        },
        verbose=1,
        tensorboard_log=log_dir,
        device="cuda"
    )'''

    # Load previous model
    model_path = "./td3_models/demo_td3"
    replay_path = "./td3_models/replay_buffer_5.pkl"

    # Rebuild environment and load model
    old_model = TD3.load(model_path, env=env, device="cuda", print_system_info=True)
    '''model.replay_buffer = ReplayBuffer(
        buffer_size=5_000_000,                # ~25% of total steps
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=model.device,
        optimize_memory_usage=False
    )'''



    # Step 2: Create a new model with desired gamma
    model = TD3(
        "MlpPolicy",
        env=env,
        gamma=0.999,  
        buffer_size=5_000_000,
        policy_kwargs=old_model.policy_kwargs,
        learning_rate=1e-5,
        tau=old_model.tau,
        train_freq=old_model.train_freq,
        policy_delay=old_model.policy_delay,
        action_noise=action_noise,
        verbose=1,
        device="cuda",
    )

    # Load replay buffer
    '''if os.path.exists(replay_path):
        model.load_replay_buffer(replay_path)
        print("[INFO] Replay buffer loaded.")
    else:
        print("[WARNING] Replay buffer not found!")'''

    # Step 3: Load weights
    model.policy.load_state_dict(old_model.policy.state_dict())
    # OPTIONAL: Re-assign callbacks or noise if needed'''
    
    model.set_logger(logger)






    print("Model Created. Training begins...")

    try:
        model.learn(
            total_timesteps=TIMESTEPS,
            callback=[eval_callback, decay_callback],
            progress_bar=True,
            reset_num_timesteps=True
        )
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted. Saving model...")
    finally:
        model.save("./td3_summer_models/td3_model_1")
        model.save_replay_buffer("./td3_summer_models/replay_buffer_large.pkl")
        print("[INFO] Model and replay buffer saved.")

