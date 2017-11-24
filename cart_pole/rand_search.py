import numpy as np
import gym
from gym import wrappers
from matplotlib import pyplot as plt

env = gym.make('CartPole-v0')
print(type(env))

best_avg = 0
best_w = np.array([ 0.17006581,  0.8227281,   0.92784342,  0.76484426])

states = []

for m in range(200):
    weights = np.random.rand(4)
    lst = []
    for k in range(50):
        obs = env.reset() #pos, vel, angle, vel at top
        done = False
        i = 0.0

        while not done:
            # env.render()
            action = 1 if weights.dot(obs.T) > 0 else 0
            obs, r, done, _ = env.step(action)
            states.append(obs)
            i+=1
        lst.append(i)
    # print(np.average(lst))

    if np.average(lst) > best_avg:
       best_w = weights
       best_avg = np.average(lst)

print(best_avg, best_w)

# env = wrappers.Monitor(env, 'records', force=True)

# obs = env.reset() #pos, vel, angle, vel at top
# done = False
# while not done:
#     # env.render()
#     action = 1 if best_w.dot(obs.T) > 0 else 0
#     obs, r, done, _ = env.step(action)

for i in range(4):
    print('N%d'%i, min(np.array(states)[:,i]), max(np.array(states)[:,i]))