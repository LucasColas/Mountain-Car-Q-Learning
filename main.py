import gym
import collections
import numpy as np


env_name = "MountainCar-v0"
env = gym.make(env_name)
#print( env.observation_space.high)
#print(env.observation_space.low)

Alpha = 0.1
Gamma = 0.9
