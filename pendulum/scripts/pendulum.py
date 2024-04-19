import gym
import sys
import gymnasium
sys.modules["gym"] = gymnasium
import numpy as np 
from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

vec_env = make_vec_env("Pendulum-v1", n_envs=1)

model = DDPG("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=20000, log_interval=10)

soil_trajectory = []
reward = []
state_trajectory = {}
reward_trajectory = {}
ACCEPTED_TRAJECTORIES = 10
for i in range(500):
    obs = vec_env.reset()
    terminated = False
    rewards_count = []
    trajectory = []
    while terminated == False:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        trajectory.append(obs[0])
        terminated = dones[0]
        rewards_count.append(rewards[0])
    reward.append(sum(rewards_count))
    state_trajectory[i] = trajectory
    reward_trajectory[i] = sum(rewards_count)
    
mean = np.mean(reward)
std = np.std(reward)
reward_threshold = mean + std
for idx in range(100):
    if reward_trajectory[idx] > reward_threshold:
        soil_trajectory.append(state_trajectory[idx])
        
np.save('Pendulum.npy', soil_trajectory)
    