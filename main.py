import gym
import collections
import numpy as np


env_name = "MountainCar-v0"
env = gym.make(env_name)
#env.reset()
#obs_, reward,done, _ = env.step(1)
#print(obs_)
print(env.observation_space.high)
print(env.observation_space.low)

Alpha = 0.1
Gamma = 0.9
Eps = 1.0

pos_chunk = np.linspace(env.observation_space.low[0], env.observation_space.high[0],20)
vel_chunk = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 20)

def get_discrete_state(state):
    pos_dis = np.digitize(state[0], pos_chunk)
    vel_dis = np.digitize(state[1], pos_chunk)

    return (pos_dis, vel_dis)

def create_Q_table():
     Q = {}
     states = []
     for i in range(len(pos_chunk)):
         for j in range(len(vel_chunk)):
             states.append((i,j))

    for state in states:
        for action in range(3):
            Q[state,action] = 0

     return Q

def get_best_action(state,Q):
    actions = np.array([Q[state,act] for act in range(3)])
    best_action = np.argmax(actions)

    return best_action

def main(env,Alpha,Gamma,Eps,n_ep=5000):
    done = False

    for i in range(n_ep):
        state = env.reset()
        state_dis = get_discrete_state(state)
        while not done:
            if np.random.random() > Eps:
                action = get_best_action(state_dis,Q)

            else:
                action = np.random.choice([0,1,2])

            new_state, reward, done, info = env.step(action)
            new_state_dis = get_discrete_state(new_state)

            new_action = get_best_action(new_state_dis, Q)

            Q[state, action] = Q[state,action] + Alpha*(reward + Gamme*Q[new_state_dis, new_action] - Q[state,action])
            state_dis = new_state_dis
