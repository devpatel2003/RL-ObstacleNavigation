import os
import pickle
import numpy as np
import torch 
import torch.nn as nn

from imitation.algorithms.adversarial.airl import AIRL
from imitation.util.util import make_vec_env
from imitation.util.networks import RunningNorm 
from imitation.data.wrappers import RolloutInfoWrapper

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv

from IRL_maze_gym import CrowdAvoidanceEnv  

print(torch.cuda.is_available())        
print(torch.cuda.get_device_name(0))    

# Create training environment with GUI
env = CrowdAvoidanceEnv()

def make_env():
    return CrowdAvoidanceEnv()


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
                nn.Linear(3, 512),  
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU()
        )

        # 2nd: CNN with BatchNorm for static obsticals https://arxiv.org/pdf/2410.07447 <- cool research
        #      (Removed built-in padding, we'll do manual circular padding instead)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=num_filters, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
            nn.Conv1d(in_channels=num_filters, out_channels=num_filters * 2, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(num_filters * 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), # Reduce each frame to a single vector
            nn.Flatten()
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
        extra_features = observations[:, :3]  # shape: (1, 7)

        # LiDAR data shape =(batch, 5 * num_readings)
        lidar_data = observations[:, 3:]
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


class RewardNet(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        obs_size = observation_space.shape[0]
        act_size = action_space.shape[0]
        self.network = nn.Sequential(
            nn.Linear(obs_size + act_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, obs, acts, next_obs=None, dones=None):
        obs =torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        acts =torch.as_tensor(acts, dtype=torch.float32, device=self.device)
        
        # Extract first 3 + min of remaining
        first_three = obs[:, :3]
        min_rest =torch.min(obs[:, 3:], dim=1, keepdim=True)[0]
        
        # Concatenate and pass through network
        x =torch.cat([first_three, min_rest, acts], dim=1)
        return self.network(x).squeeze(dim=1)


    def predict_processed(self, obs, acts, next_obs=None, dones=None):
        with torch.no_grad():
            rew = self.forward(obs, acts)
        return rew.cpu().numpy()
    
    def preprocess(self, obs, acts, next_obs, dones):
        obs_torch = torch.tensor(obs, dtype=torch.float32)
        acts_torch = torch.tensor(acts, dtype=torch.float32)
        next_obs_torch = torch.tensor(next_obs, dtype=torch.float32)
        dones_torch = torch.tensor(dones, dtype=torch.float32)
        return obs_torch, acts_torch, next_obs_torch, dones_torch
    
    @property
    def device(self):
        return next(self.parameters()).device



# === Load expert trajectories ===
def load_trajectories(directory, limit=None):
    demos = []
    files = sorted(os.listdir(directory))[:limit]
    for fname in files:
        with open(os.path.join(directory, fname), "rb") as f:
            trajectory = pickle.load(f)  
            demos.append(trajectory)   
    return demos 


# === Create env ===
venv = DummyVecEnv([make_env])
venv = RolloutInfoWrapper(venv)

# === Load expert data ===
expert_demos = load_trajectories("expert_trajectories3", limit=200)

# === Set up AIRL ===
reward_net = RewardNet(
    observation_space=venv.observation_space,
    action_space=venv.action_space,
).to("cuda")

# Create logging directory
log_dir = "airl_logs"
os.makedirs(log_dir, exist_ok=True)

ppo_model = PPO(
        "MlpPolicy",
        env,
        learning_rate=2e-4,       # Learning rate (default: 3e-4)
        batch_size=512,           # Batch size for training     
        clip_range=0.2,
        ent_coef=0.02, 
        n_steps = 2048,# Train after every 5s
        policy_kwargs = {
            "net_arch": [],
            "features_extractor_class": CustomCNNLSTM,
            "features_extractor_kwargs": {
                "num_filters": 64,
                "lstm_hidden_size": 256,
                "train_cnn": True,
                "train_lstm": False,      
                "train_mlp": True          
            },
        },
        verbose=1,                # Logging level (1 = info, 0 = silent)
        tensorboard_log= log_dir,
        device = "cuda"
)

# Configure SB3 logger
airl_logger = configure(log_dir, ["stdout", "tensorboard"])




airl_trainer = AIRL(
    demonstrations=expert_demos,
    demo_batch_size=64,
    n_disc_updates_per_round=4,
    gen_algo= ppo_model,
    reward_net=reward_net,
    venv=venv,
    allow_variable_horizon=True,
)

airl_trainer.gen_algo.set_logger(airl_logger)


# === Train AIRL ===
airl_trainer.train(total_timesteps=20_000_000)

# === Save policy and reward ===
airl_trainer.gen_algo.save("airl_models/airl_policy")
torch.save(reward_net.state_dict(), "airl_models/learned_reward.pt")
print("AIRL training complete!")