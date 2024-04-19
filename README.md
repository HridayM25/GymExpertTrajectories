# Expert Trajectories for Gym Environments
Contains expert trajectories tailored for state-only imitation learning tasks in Gym environments. 

## What's State-Only Imitation Learning?
State-only imitation learning is all about mimicking the behavior of an expert using only the observations of the environment's states. It's like learning to dance by watching someone else's moves without knowing the exact steps/actions they are taking.

## How to Use 
In the `trajectories` directory, you will find the trajectories. Use `np.load` to use the trajectories for SOIL.

## Methods of Collection 

| Environment | Algorithm   | Details   | Trajectories Collected |
| :---:   | :---: | :---: | :---:|
| CartPole | PPO   | I have collected trajectories which yielded a minimum reward of 475. This can be found in the official documentation of the environment.   | 1000|
| Pendulum  | DDPG | Pendulum is an unsolved environment. Hence the approach I have taken is first collect 1e6 trajectories of the trained agent. I have then taken the values that lie greater than `mean + 2 * standard_deviation` of the rewards. I have sampled those trajectories only. | 22|
| MountainCar Continuous | DDPG   | I have collected trajectories which yielded a minimum reward of -110.   | 1000|
