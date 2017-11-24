import os
import sys
import gym
import matplotlib
import collections
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from gym.wrappers.time_limit import TimeLimit


class BaseModel:

    def __init__(self, D, l, gamma):
        self.w = np.random.random((1,D))/np.sqrt(D)
        self.e = np.zeros((1,D))
        self.lr = 0.01
        self.l = l
        self.gamma = gamma

    def partial_fit(self, X, G):
        diff = G - self.predict(X)
        self.e = self.e*self.l*self.gamma + X
        self.w += self.lr*diff*self.e

    def predict(self, X):
        return self.w.dot(X.T)


class FeatureTransformer:
    def __init__(self, env:TimeLimit):
        observation_examples = np.array([env.observation_space.sample() for _ in range(10000)])
        scaler = StandardScaler()
        scaler.fit(observation_examples)


        featurizer = FeatureUnion([
            ('rbf1', RBFSampler(gamma=5.0, n_components=500)),
            ('rbf2', RBFSampler(gamma=2.0, n_components=500)),
            ('rbf3', RBFSampler(gamma=1.0, n_components=500)),
            ('rbf4', RBFSampler(gamma=0.5, n_components=500)),
        ])

        featurizer.fit(scaler.transform(observation_examples))
        self.scaler = scaler
        self.featurizer = featurizer

    def transform(self, obs):
        scaled = self.scaler.transform(obs)
        return self.featurizer.transform(scaled)

class Model:
    def __init__(self, env, feature_transformer:FeatureTransformer, learning_rate):
        self.env = env
        self.models = [] # type: list[BaseModel]
        self.ft = feature_transformer
        for i in range(env.action_space.n):
            model = BaseModel(2000, 0.7, 0.99)
            self.models.append(model)

    def predict(self, s):
        x = self.ft.transform([s])
        assert (len(x.shape)==2)
        return np.array([m.predict(x)[0] for m in self.models])
        # return list(map(lambda model, x: model.predict(x)[0], self.models, [x]*3))

    def update(self, s, a, G):
        x = self.ft.transform(np.atleast_2d(s))
        self.models[a].partial_fit(x, G)

    def sample_action(self, s, eps=0):
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            pred = self.predict(s)
            return np.argmax(pred)




def play_one(model, n, eps=0, gamma=0.99, l=0.5):
    obs = model.env.reset()
    done = False
    iters = 0
    total_rew = 0

    e = np.zeros(2000)

    while not done:
        a = model.sample_action(obs, eps)
        prev_obs = obs
        obs, rew, done, info = model.env.step(a)

        total_rew += rew

        G = rew + gamma * np.max(model.predict(obs)[0])
        model.update(prev_obs, a, G)

        iters += 1

    return total_rew

def plot_cost_to_go(env, estimator, num_tiles=20):
    x = np.linspace(env.observation_space.low[0], env.observation_space.high[0])
    y = np.linspace(env.observation_space.low[1], env.observation_space.high[1])
    X, Y = np.meshgrid(x,y)

    Z = np.apply_along_axis(lambda _: -np.max(estimator.predict(_)), 2, np.dstack([X, Y]))

    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X,Y,Z, rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Cost-to-go == -V(s)')
    ax.set_title('Cost-to-go-function')
    fig.colorbar(surf)
    plt.show()


def plot_running_avg(totalrewards:np.ndarray):
    N = len(totalrewards)
    running_avg = np.empty(N)
    for i in range(N):
        running_avg[i] = totalrewards[max(0,i-100):(i+1)].mean()
    plt.plot(running_avg)
    plt.show()



if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    ft = FeatureTransformer(env)
    model = Model(env, ft, 'constant')

    N = 300
    totalrewards = np.empty(N)
    gamma = 0.99

    for i in range(N):
        eps = 0.1*(0.97**i)
        totalreward = play_one(model, 7, eps=eps, gamma=gamma)
        totalrewards[i] = totalreward
        print("episode:", i, "total reward:", totalreward, "eps:", eps)
    print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
    print("total steps:", totalrewards.sum())

    plt.plot(totalrewards)
    plt.title('Rewards')
    plt.show()

    plot_running_avg(totalrewards)

    plot_cost_to_go(env, model)