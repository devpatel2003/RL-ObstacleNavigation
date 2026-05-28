import gymnasium as gym
import stable_baselines3 as sb3
import sb3_contrib as sb3c
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from datetime import datetime

from gymnasium import Wrapper
from collections import deque
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.running_mean_std import RunningMeanStd


import torch.nn as nn
from rppo_gym import CrowdAvoidanceEnv
import time  # Needed for rendering
import pybullet as p
import shutil
import os
import torch
import numpy as np
import time
import random

random.seed(0); np.random.seed(0); torch.manual_seed(0)

torch.backends.cudnn.benchmark = True  # Optimize CUDA performance

TIMESTEPS = 10_000_000 #500 ts = 1s
MODEL_LOAD_PATH = "./new_rppo_models/best_model_v3"  # Adjust path to your model v3 is good for no obstacles
MODEL_SAVE_PATH = "./new_rppo_models/v4.zip"
ENV_LOAD_PATH =  ""#./rppo_models/vecnormalize_easy_1024_512_v1.pkl" #<-- if changing reward create new env, otherwise load old env
ENV_SAVE_PATH = "./new_rppo_models/v4.pkl"

def linear_schedule(initial: float, final: float = 0.0):
    """SB3‑style LR schedule."""
    def _lr(progress_remaining: float) -> float:
        return final + (initial - final) * progress_remaining
    return _lr

# Create training environment with GUI
def make_env(deterministic=False):
    def _init():
        env = CrowdAvoidanceEnv()
        env = Monitor(
            env,
            info_keywords=("is_success", "is_collision"))  
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

class SuccessRateCallback(BaseCallback):
    def __init__(self, window=100, log_collisions=True, verbose=0):
        super().__init__(verbose)
        self.window = window
        self.log_collisions = log_collisions
        self.success_buf = deque(maxlen=window)
        self.collision_buf = deque(maxlen=window)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            # Only count completed episodes
            if "episode" in info:
                if "is_success" in info:
                    self.success_buf.append(int(info["is_success"]))
                if self.log_collisions and "is_collision" in info:
                    self.collision_buf.append(int(info["is_collision"]))

        if self.success_buf:
            self.logger.record(
                f"rollout/success_rate_last_{self.window}",
                float(np.mean(self.success_buf))
            )
        if self.log_collisions and self.collision_buf:
            self.logger.record(
                f"rollout/collision_rate_last_{self.window}",
                float(np.mean(self.collision_buf))
            )

        return True


if __name__ == "__main__":

    print("CUDA Available:", torch.cuda.is_available())
    print("Number of GPUs:", torch.cuda.device_count())
    print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")
    print("Torch Version:", torch.__version__)
    print("Torch CUDA Version:", torch.version.cuda)
    print("SB3 Version:", sb3.__version__)
    print("SB3-Contrib Version:", sb3c.__version__)

    # Ensure fresh logs before training
    log_dir = f"./new_rppo_logs/continued_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(log_dir, exist_ok=True)
    logger = configure(log_dir, ["stdout", "tensorboard"])

    n_envs = 10  
    env = SubprocVecEnv([make_env() for _ in range(n_envs)])
    eval_env = DummyVecEnv([make_env()])

    # VecNormalize setup
    '''if os.path.exists(ENV_LOAD_PATH):
        env = VecNormalize.load(ENV_LOAD_PATH, env)
        env.training = True            # keep updating running stats
        env.norm_reward = True         # if you still want reward norm

        eval_env = VecNormalize.load(ENV_LOAD_PATH, eval_env)
        eval_env.training = False       # disable updates during evaluation
        eval_env.norm_reward = False  # no reward norm during eval
        print(f"Loaded VecNormalize stats from {ENV_LOAD_PATH}")
    else:
        env = VecNormalize(
            env,
            norm_obs=False,          # keep your engineered obs scaling
            norm_reward=True,        # normalize returns -> stabler updates
            clip_reward=100.0,        # clip normalized returns; tune if needed
            clip_obs=10.0            # unused when norm_obs=False, but must pass
        )
        env.training = True
        env.norm_reward = True
        env.return_rms = RunningMeanStd(shape=())

          # Set up evaluation callback
        eval_env = VecNormalize(
            eval_env,
            norm_obs=False,
            norm_reward=False,
            clip_reward=10.0,
            clip_obs=10.0,
            training=False  # disable updates during evaluation
        )
        print("Created new VecNormalize environment.")'''

  

    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path="./new_rppo_models/",
                                log_path=log_dir, 
                                eval_freq=2_000  , 
                                deterministic=True, 
                                render=False
        )
    success_cb = SuccessRateCallback(window=100, log_collisions=True)
    callback = CallbackList([eval_callback, success_cb])

    model = RecurrentPPO(
        "MlpLstmPolicy",          # built‑in LSTM policy
        env,
        #learning_rate  = 2e-4,    # PPO LR, a bit higher than TD3
        learning_rate  = 2e-4,#linear_schedule(2e-4, 1e-5),  # PPO LR, a bit higher than TD3
        n_steps        = 1024,     # rollout length per env (×12 envs = 1536 steps/iter)
        batch_size     = 512,     # minibatch for SGD
        gamma          = 0.99,
        gae_lambda     = 0.95,
        clip_range     = 0.2,
        vf_coef        = 0.5,
        ent_coef       = 0.001,   # keeps exploration alive
        tensorboard_log= log_dir,
        policy_kwargs  = dict(
            net_arch=[1024, 1024],  # MLP before LSTM (then hidden 256‑unit LSTM),  # MLP before LSTM (then hidden 256‑unit LSTM)
            lstm_hidden_size=256,  # LSTM hidden size
            activation_fn=torch.nn.ReLU
        ),
        verbose        = 1,
        device         = "cuda"
    )

    print(model.lr_schedule(1.0))   # 2e-4  (start)
    print(model.lr_schedule(0.5))   # midway value
    print(model.lr_schedule(0.0))   # 1e-5  (end)

    model = RecurrentPPO.load(MODEL_LOAD_PATH, env=env, device="cuda")
       
    #model.policy.optimizer = torch.optim.Adam(model.policy.parameters(), lr=1e-4)
    #model.lr_schedule = lambda _: 1e-4




    model.set_logger(logger)



    
    print("Model Created. Training begins...")


    try:
        model.learn(
            total_timesteps=TIMESTEPS,
            callback=callback,
            progress_bar=True,
            reset_num_timesteps=False,
            tb_log_name="continued_RPPO",
        )
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted. Saving model...")
    finally:
        model.save(MODEL_SAVE_PATH)
        #env.save(ENV_SAVE_PATH)
        print(f"[INFO] Saved VecNormalize stats to {ENV_SAVE_PATH}")
        print("[INFO] Model and replay buffer saved.")

    # Save final trained model


    #1 just target
    #2 lidar added
    #3 cnn added