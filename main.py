import gym
import collections
import numpy as np


env_name = "MountainCar-v0"
env = gym.make(env_name)
env.reset()
obs_, reward,done, _ = env.step(1)
print(obs_)
#print( env.observation_space.high)
#print(env.observation_space.low)

Alpha = 0.1
Gamma = 0.9
Eps = 1.0

def create_Q_table():
     pass 
