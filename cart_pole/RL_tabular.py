import numpy as np
import gym
from gym import wrappers
from matplotlib import pyplot
import time
from tutor_files import q_learning_bins as ql


GAMMA = 0.9
ALPHA = 0.01

env = gym.make('CartPole-v0')
ft = ql.FeatureTransformer()
# env = wrappers.Monitor(env, 'records')

res = np.array([5,5,5,5])
off = np.array([2.4,3.2,0.4,4])
aux = [1]

for i in range(3):
    aux.append(res[i]*2*aux[i])

def state_to_bin(state):
    state = np.array([state[i] if abs(state[i]) < off[i] - off[i]/res[i] else off[i] * np.sign(state[i]) for i in range(4)])
    ts = state + off
    n = round(ts[0])
    for i in range(1,4):
        n += int(round(ts[i])*aux[i])
    r = ft.transform(state)
    return r


def random_action(a, eps):
    if np.random.random() < eps:
        return env.action_space.sample()
    else:
        return a


def make_step(a, env=env):
    obs, r, done, _ = env.step(a)
    if done and env._elapsed_steps < 199:
        return -300, state_to_bin(obs), done
    else:
        return 0, state_to_bin(obs), done


def Sarsa():
    Q = {}
    graph = []

    N = 10000
    for i in range(N):
        eps = 1.0/np.sqrt((i+1))
        s = env.reset()
        s = state_to_bin(s)
        a = np.argmax([Q.get(s, {}).get(a, 0) for a in [0,1]])
        a = random_action(a, eps)
        done = False

        while not done:
            r, sp, done = make_step(a)
            ap = np.argmax([Q.get(s, {}).get(a, 0) for a in [0,1]])
            ap = random_action(ap,eps)
            if s not in Q:
                Q[s] = {}
            G = np.max([Q.get(sp, {}).get(ap, 0) for a in [0,1]])
            Q[s][a] = Q.get(s, {}).get(a, 0) + ALPHA*(r + G - Q.get(s, {}).get(a, 0))
            s = sp
            a = ap

        graph.append(env._elapsed_steps)

    env2 = wrappers.Monitor(env, 'records', force=True)
    s = env.reset()
    s = state_to_bin(s)
    a = np.argmax([Q.get(s, {}).get(a, 0) for a in [0, 1]])

    done = False

    while not done:
        env2.render()
        r, sp, done = make_step(a)

        ap = np.argmax([Q.get(s, {}).get(a, 0) for a in [0, 1]])
        time.sleep(0.1)

    graph = moving_average(graph,100)
    pyplot.plot(graph)
    pyplot.show()

    print(Q)


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

if __name__ == '__main__':
    Sarsa()