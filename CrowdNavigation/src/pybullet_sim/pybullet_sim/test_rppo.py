# test_rppo.py  ------------------------------------------------------------
import torch, numpy as np, gymnasium as gym
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from rppo_gym import CrowdAvoidanceEnv        # <- your env class
import time  # Needed for rendering
import os

MODEL_PATH       = "./new_rppo_models/best_model.zip"
#MODEL_PATH     = "./rppo_models/big_obs_rppo_model_1024.zip"  # Adjust path to your model
NORMALIZE_PATH   =""# "./new_rppo_logs/vecnormalize_1024.pkl"  # optional
N_EPISODES       = 10
RENDER           = True            # Pop a PyBullet GUI window

# -------------------------------------------------------------------------
# 1) Build a *single* env (optionally with GUI)
def make_env_gui():
    return CrowdAvoidanceEnv(use_gui=RENDER)

# 2) Restore VecNormalize if it was used during training
if os.path.exists(NORMALIZE_PATH):
    dummy_env = DummyVecEnv([make_env_gui])           # needs a dummy
    env = VecNormalize.load(NORMALIZE_PATH, dummy_env)
    env.training = False                              # disable updates
    env.norm_reward = False
else:
    env = DummyVecEnv([make_env_gui])

# 3) Load the trained RecurrentPPO policy
model = RecurrentPPO.load(MODEL_PATH, env=env, device="cuda")

# -------------------------------------------------------------------------
success, collision, returns = 0, 0, []

for ep in range(N_EPISODES):
    obs = env.reset()
    # Important for recurrent policies ↓
    lstm_state = None                               # model will create zeros
    episode_return, done = 0.0, False
    while not done:
        action, lstm_state = model.predict(
            obs,
            state=lstm_state,
            episode_start=np.array([done]),
            deterministic=True,     # ← set False to watch exploration
        )
        obs, reward, done, trunc = env.step(action)
    
        episode_return += reward[0]
        if RENDER:                         # show at real time speed
            time.sleep(env.envs[0].CTRL_DT)

    # bookkeeping ---------------------------------------------------------
    returns.append(episode_return)

    print(f"Episode {ep+1:02d} | return {episode_return:6.1f}")

env.close()

print("\n==========  Test summary  ==========")
print(f"Success rate   : {success}/{N_EPISODES}")
print(f"Collision rate : {collision}/{N_EPISODES}")
print(f"Mean return    : {np.mean(returns):.1f}")
print("====================================")
