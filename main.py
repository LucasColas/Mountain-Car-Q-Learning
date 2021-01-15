import gym
import collections
import numpy as np


env_name = "MountainCar-v0"
env = gym.make(env_name)
#print( env.observation_space.high)
#print(env.observation_space.low)

Alpha = 0.1
Gamma = 0.9

def create_Q_table(env):
    ObservationSpace_Size = [30] * len(env.observation_space.low)
    Q_table = np.random.uniform(low=-2, high=0, size=(ObservationSpace_Size + [env.action_space.n]))
    return Q_table

env.reset()
Q_table = create_Q_table(env)
print(Q_table)

class Agent:
    def __init__(env):
        self.env = env
