# https://deeplearningcourses.com/c/deep-reinforcement-learning-in-python
# https://www.udemy.com/deep-reinforcement-learning-in-python
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future
#
# Note: gym changed from version 0.7.3 to 0.8.0
# MountainCar episode length is capped at 200 in later versions.
# This means your agent can't learn as much in the earlier episodes
# since they are no longer as long.
#
# Adapt Q-Learning script to use TD(lambda) method instead

import gym
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime

# code we already wrote
from mountain_car.mount_car import plot_cost_to_go, FeatureTransformer, plot_running_avg


class BaseModel:
  def __init__(self, D, g, l):
    self.w = np.random.randn(D) / np.sqrt(D)
    self.e = np.zeros(D)
    self.gamma = g
    self.l = l

  def partial_fit(self, input_, target, lr=1e-2):
    self.e *= self.gamma * self.l
    self.e += input_
    self.w += lr*(target - input_.dot(self.w))*self.e

  def predict(self, X):
    X = np.array(X)
    return X.dot(self.w)


# Holds one BaseModel for each action
class Model:
  def __init__(self, env, feature_transformer, gamma,lambda_):
    self.env = env
    self.models = []
    self.feature_transformer = feature_transformer

    D = 2000
    self.eligibilities = np.zeros((env.action_space.n, D))
    for i in range(env.action_space.n):
      model = BaseModel(D, gamma, lambda_)
      self.models.append(model)

  def predict(self, s):
    X = self.feature_transformer.transform([s])
    assert(len(X.shape) == 2)
    return np.array([m.predict(X)[0] for m in self.models])

  def update(self, s, a, G):
    X = self.feature_transformer.transform([s])
    assert(len(X.shape) == 2)
    self.models[a].partial_fit(X[0], G)

  def sample_action(self, s, eps):
    if np.random.random() < eps:
      return self.env.action_space.sample()
    else:
      return np.argmax(self.predict(s))


# returns a list of states_and_rewards, and the total reward
def play_one(model, eps):
  observation = env.reset()
  done = False
  totalreward = 0
  iters = 0
  # while not done and iters < 200:
  while not done and iters < 10000:
    action = model.sample_action(observation, eps)
    prev_observation = observation
    observation, reward, done, info = env.step(action)

    # update the model
    G = reward + gamma*np.max(model.predict(observation)[0])
    model.update(prev_observation, action, G)

    totalreward += reward
    iters += 1

  return totalreward


if __name__ == '__main__':
  env = gym.make('MountainCar-v0')
  ft = FeatureTransformer(env)
  gamma = 0.99
  lambda_ = 0.7
  model = Model(env, ft, gamma, lambda_)

  if 'monitor' in sys.argv:
    filename = os.path.basename(__file__).split('.')[0]
    monitor_dir = './' + filename + '_' + str(datetime.now())
    env = wrappers.Monitor(env, monitor_dir)


  N = 300
  totalrewards = np.empty(N)
  costs = np.empty(N)
  for n in range(N):
    # eps = 1.0/(0.1*n+1)
    eps = 0.1*(0.97**n)
    # eps = 0.5/np.sqrt(n+1)
    totalreward = play_one(model, eps)
    totalrewards[n] = totalreward
    print("episode:", n, "total reward:", totalreward)
  print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
  print("total steps:", -totalrewards.sum())

  plt.plot(totalrewards)
  plt.title("Rewards")
  plt.show()

  plot_running_avg(totalrewards)

  # plot the optimal state-value function
  plot_cost_to_go(env, model)
