import gym
import sys
import gymnasium
sys.modules["gym"] = gymnasium
import numpy as np 
from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

vec_env = make_vec_env("MountainCarContinuous-v0", n_envs=1)
n_actions = vec_env.action_space.n
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
model = DDPG("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=20000, log_interval=10)

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
    if sum(rewards_count) >= -110: #cartonpole-v1 env solving reward
        soil_trajectory.append(trajectory)
    if len(soil_trajectory) == ACCEPTED_TRAJECTORIES:
        break

np.save('MountainCarTrajectory.npy', soil_trajectory)