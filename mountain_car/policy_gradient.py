import os
import sys
import gym
import matplotlib
import collections
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf

import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from gym.wrappers.time_limit import TimeLimit

from mountain_car.mount_car_TD_lambda import FeatureTransformer, plot_running_avg, plot_cost_to_go

class Model:
    def __init__(self, D, n_action):
        # Layer's sizes
        # Symbols
        self.G = tf.placeholder("float", shape=[None, 1])
        self.state = tf.placeholder("float", shape=[None, D])
        self.a = tf.placeholder("int32", shape=[None, 2])
        self.adv = tf.placeholder("float", shape=[None, 1])

        # theta initializations
        self.theta_P = tf.Variable(tf.random_normal((n_action,D), stddev=0.1))
        self.theta_V = tf.Variable(tf.random_normal((1,D), stddev=0.1))

        # pi(action|state,theta) [None x 3]
        self.yhat_policies = tf.nn.softmax(tf.matmul(self.state, self.theta_P, transpose_b=True))
        # pi(action|state,theta) [1 x None]
        self.yhat_policy = tf.expand_dims(tf.gather_nd(self.yhat_policies, self.a), 0)

        self.yhat_value = tf.matmul(self.state, self.theta_V, transpose_b=True)

        #cost_P [1 x 1] = [1 x None] * [None x 1]
        self.cost_P = tf.matmul(tf.log(self.yhat_policy), self.adv)
        self.update_theta_P = tf.train.GradientDescentOptimizer(0.01).minimize(-self.cost_P)
        #labels=[None x 1] logits=[None x 1])
        cost_V = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.G, logits=self.yhat_value))
        self.update_theta_V = tf.train.GradientDescentOptimizer(0.01).minimize(cost_V)

        # Run SGD
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def update(self, X, adv, G, a):
        p = self.sess.run(self.yhat_policy, {self.state: X, self.G: G, self.a:a})
        cost = self.sess.run(self.cost_P, {self.state: X, self.G: G, self.adv:adv, self.a:a})
        aaa = self.sess.run(self.update_theta_V, {self.state:X, self.G:G})
        thetaV = self.theta_V.eval(session=self.sess)

        bbb= self.sess.run(self.update_theta_P, {self.state:X, self.G:G, self.adv:adv, self.a:a})
        thetaP = self.theta_P.eval(session=self.sess)
        print('')

    def predict_pi(self, X, a):
        p = self.sess.run(self.yhat_policies, {self.state: X, self.a: a})
        thetaV = self.theta_V.eval(session=self.sess)
        thetaP = self.theta_P.eval(session=self.sess)
        pi = self.sess.run(self.yhat_policy, {self.state:X, self.a:a})
        return pi

    def predict_V(self,X ):
        return self.sess.run(self.yhat_value, {self.state:X})

class Model_s:
    def __init__(self, env, feature_transformer:FeatureTransformer, learning_rate):
        self.env = env
        self.model = Model(2000, env.action_space.n)
        self.ft = feature_transformer

    def predict_pi(self, s, a):
        x = self.ft.transform(np.atleast_2d(s))
        return self.model.predict_pi(x,a)
        # return list(map(lambda model, x: model.predict(x)[0], self.models, [x]*3))

    def predict_V(self, s):
        x = self.ft.transform(np.atleast_2d(s))
        return self.model.predict_V(x)

    def update(self, s, adv, a, G):
        x = self.ft.transform(np.atleast_2d(s))
        if type(a) == 'int':
            act = np.array([[0, a]])
        else:
            act = np.hstack((np.arange(0,len(a))[np.newaxis].T, a[np.newaxis].T))
        self.model.update(x, adv, G, act)

    def sample_action(self, s, eps=0):
        pred = np.squeeze([self.predict_pi(s, np.array([[0, a]])) for a in range(3)])
        # print(pred)
        return np.random.choice(3, 1, p=pred)[0]


def play_one(model, n, eps=0, gamma=0.99, l=0.5):
    obs = model.env.reset()
    done = False
    iters = 0
    total_rew = 0

    e = np.zeros(2000)

    Gs = []
    As = []
    POs = []
    while not done:
        # model.env.render()
        a = model.sample_action(obs, eps)
        prev_obs = obs
        obs, rew, done, info = model.env.step(a)

        total_rew += rew

        if done and model.env._elapsed_steps < 199:
            rew = -200

        G = rew + gamma * np.max(model.predict_V(obs)[0])
        Gs.append([rew])
        As.append(a)
        POs.append(prev_obs)
        iters += 1

    Gs = np.array(Gs)
    Gs = np.array([np.sum(Gs[:,i:]) for i in range(len(Gs))])
    pred = model.predict_V(POs)
    advantage = Gs[np.newaxis].T - pred

    model.update(np.array(POs), advantage, np.array(As), np.array(Gs)[np.newaxis].T)

    return total_rew


if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    ft = FeatureTransformer(env)
    model = Model_s(env, ft, 'constant')

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