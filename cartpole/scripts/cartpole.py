import gym
import numpy as np 
import json 
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


vec_env = make_vec_env("CartPole-v1", n_envs=1)
model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=25000)
obs = vec_env.reset()

#You can change the number of accepted trajectories to modify your dataset

soil_trajectory = []
ACCEPTED_TRAJECTORIES = 1000
for i in range(10000):
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
    if sum(rewards_count) > 475: #cartonpole-v1 env solving reward
        soil_trajectory.append(trajectory)
    if len(soil_trajectory) == ACCEPTED_TRAJECTORIES:
        break
        
np.save('cartpoleTrajectory.npy', soil_trajectory)