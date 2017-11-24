import numpy as np
import gym
import time

env = gym.make('CartPole-v0')
lst = []
ang = []
for k in range(10000):
    state = env.reset() #pos, vel, angle, vel at top
    done = False
    i = 0.0
    while not done:
        # env.render()
        action = env.action_space.sample()
        obs, r, done, _ = env.step(0)
        i+=1
        # time.sleep(0.1)
    lst.append(i)
    ang.append(obs[2])

print(np.average(lst))
print(np.average(ang))