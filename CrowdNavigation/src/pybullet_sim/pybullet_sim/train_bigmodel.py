import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
from static_gym import CrowdAvoidanceEnv
import time  # Needed for rendering
import pybullet as p
import shutil
import os
import torch
import numpy as np


torch.backends.cudnn.benchmark = True  # Optimize CUDA performance

TIMESTEPS = 20_000_000 #500 ts = 1s

def circular_pad_1d(x, pad=2):
    """
    x shape: (batch_size, channels, length)
    use circular padding on ring of lidar scans, to ensure first and last elements wrap around
    """
    left_pad = x[..., -pad:]  # last 'pad' angles
    right_pad = x[..., :pad]  # first 'pad' angles
    return torch.cat([left_pad, x, right_pad], dim=-1)

class CustomCNNLSTM(BaseFeaturesExtractor):
    def __init__(self, observation_space, num_filters=64, lstm_hidden_size=256, train_cnn=True, train_lstm=True, train_mlp=False):
        super(CustomCNNLSTM, self).__init__(observation_space, features_dim=512)

        self.train_lstm = train_lstm

        # 1st: Define MLP for goal-seeking
        self.mlp_fc = nn.Sequential(
                nn.Linear(7, 512),  
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU()
        )

        # 2nd: CNN with BatchNorm for static obsticals https://arxiv.org/pdf/2410.07447 <- cool research
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=num_filters, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
            nn.Conv1d(in_channels=num_filters, out_channels=num_filters * 2, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(num_filters * 2),
            nn.ReLU(),
            #nn.AdaptiveAvgPool1d(1), # Reduce each frame to a single vector
            #nn.Flatten()
        )

        # 3rd: LSTM for dynamic obstacle tracking
        self.lstm = nn.LSTM(
            input_size=num_filters * 2, 
            hidden_size=lstm_hidden_size,
            batch_first=True)
        
        self.dummy_linear = nn.Linear(num_filters * 2, lstm_hidden_size)
        
        # 4th: Linear layer 
        self.linear = nn.Linear(lstm_hidden_size + 512, 512) # Lstm ouput + 512 weights from mlp

        # Freeze the MLP:
        if not train_mlp:
            for param in self.mlp_fc.parameters():
                param.requires_grad = False

        # "Freeze" CNN if `train_cnn=False`
        if not train_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False  # Prevents CNN updates

        # "Freeze" LSTM if `train_lstm=False`
        if not train_lstm:
            for param in self.lstm.parameters():
                param.requires_grad = False  # Prevents LSTM updates

        
    def forward(self, observations):

        batch_size = observations.shape[0] # 1
        
        # Separate the 7 imu features
        extra_features = observations[:, :7]  # shape: (1, 7)

        # LiDAR data shape =(batch, 5 * num_readings)
        lidar_data = observations[:, 7:]
        total_lidar = lidar_data.shape[1]
        t = 5
        n = total_lidar // t  # number of readings per frame

        # Reshape to (5, 1 , 72)
        lidar_data = lidar_data.reshape(batch_size * t, 1, n)

        # Manual circular padding before passing to CNN
        lidar_data = circular_pad_1d(lidar_data, pad=2)

        # Apply CNN frame-by-frame (in parallel)
        # Reshape to (5, 1, 72)
        cnn_feats = self.cnn(lidar_data) 



        # Use dummy layer instead of lstm
        if self.train_lstm:
            # Reshape back and feed into LSTM: (1, 5, num_filters*2)
            cnn_feats = cnn_feats.reshape(batch_size, t, -1)
            lstm_out, _ = self.lstm(cnn_feats)
            # Take the last time-step's hidden state: (batch, lstm_hidden_size)
            lstm_features = lstm_out[:, -1, :]
        else:
            # Dummy path, but do the same reshape
            cnn_feats = cnn_feats.reshape(batch_size, t, -1)  # [1, 5, 128]
            dummy_feats = cnn_feats[:, -1, :]  # [1, 128]
            lstm_features = self.dummy_linear(dummy_feats)  # [1, 256]


        # Pass the 7 scalars through the frozen MLP, shape (batch, 512)
        mlp_features = self.mlp_fc(extra_features)

        # Combine only MLP output + LSTM output, shape: (batch, 512 + lstm_hidden_size)
        combined = torch.cat([mlp_features, lstm_features], dim=1)

        # Final linear, shape (batch, 512)
        output = self.linear(combined)
        return output


# Create training environment with GUI
env = CrowdAvoidanceEnv()


# Enable TensorBoard logging
log_dir = "./big_model_logs/"
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)  # Deletes previous logs
os.makedirs(log_dir, exist_ok=True)  # Creates a fresh log directory
logger = configure(log_dir, ["stdout", "tensorboard"])

# Set up evaluation callback
eval_env = CrowdAvoidanceEnv()
eval_callback = EvalCallback(eval_env, best_model_save_path="./big_models/",
                             log_path=log_dir, eval_freq=TIMESTEPS * 0.1)


# Define SAC model with proper parameters
'''model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,       # Learning rate (default: 3e-4)
        batch_size=512,           # Batch size for training     
        clip_range=0.2,
        ent_coef=0.02, 
        n_steps = 1024,# Train after every 5s
        policy_kwargs = {
            "net_arch": [],
            "features_extractor_class": CustomCNNLSTM,
            "features_extractor_kwargs": {
                "num_filters": 64,
                "lstm_hidden_size": 256,
                "train_cnn": True,
                "train_lstm": False,      
                "train_mlp": False          
            },
        },
        verbose=1,                # Logging level (1 = info, 0 = silent)
        tensorboard_log= log_dir,
        device = "cuda"
)'''

'''model = SAC(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    batch_size=512,
    buffer_size=1_000_000,  # Off-policy methods need a large replay
    tau=0.005,
    gamma=0.99,
    ent_coef="auto_0.2",  # or a fixed float
    train_freq=(1024, "step"),  # or whatever frequency you'd like
    gradient_steps=64,        # number of gradient updates after each train_freq
    policy_kwargs={
        "net_arch": [],
        "features_extractor_class": CustomCNNLSTM,
        "features_extractor_kwargs": {
            "num_filters": 64,
            "lstm_hidden_size": 256,
            "train_cnn": True,
            "train_lstm": False, 
            "train_mlp": False
        },
    },
    verbose=1,
    tensorboard_log=log_dir,
    device="cuda"
)'''

model = TD3(
    "MlpPolicy",
    env,
    learning_rate=3e-4,         # Learning rate
    batch_size=512,             # Batch size
    buffer_size=1_000_000,      # Experience replay buffer
    tau=0.005,                  # Target smoothing coefficient
    gamma=0.99,                 # Discount factor
    train_freq=(1, "episode"),  # Delay updates until new episode
    gradient_steps=100,         # Gradient updates per training iteration
    policy_kwargs={
        "net_arch": [],
        "features_extractor_class": CustomCNNLSTM,
        "features_extractor_kwargs": {
            "num_filters": 64,
            "lstm_hidden_size": 256,
            "train_cnn": False,
            "train_lstm": False,      
            "train_mlp": True          
        },
    },
    verbose=1,  # Logging level (1 = info)
    tensorboard_log=log_dir,
    device="cuda"
)

print("CUDA Available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")

# Load previous model
'''old_model_path = "./ppo_models/ppo_goal_nav"  # Adjust path
old_model = PPO.load(old_model_path, print_system_info=True)


# Load the old model's state_dict
old_state_dict = old_model.policy.state_dict()
new_state_dict = model.policy.state_dict()

for key in old_state_dict:
    # If the old key exists in the new model AND shapes match,
    # we copy it over. If they don't match or we want to skip, we skip.
    if key in new_state_dict and old_state_dict[key].shape == new_state_dict[key].shape:
        new_state_dict[key] = old_state_dict[key]

# Load the modified state dict
model.policy.load_state_dict(new_state_dict, strict=False)'''
model.set_logger(logger)

#model = SAC.load("src/pybullet_sim/pybullet_sim/sac_models/sac_crowd_avoidance_1_50k", env=env)


print("Model Created. Training begins...")


model.learn(total_timesteps=TIMESTEPS, callback=eval_callback, progress_bar=True, reset_num_timesteps=True)

# Save final trained model
model.save("./big_models/full_model_1")
print("Training Complete! Model Saved.")

#1 just target
#2 lidar added
#3 cnn added