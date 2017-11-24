import gym
import tensorflow as tf
import numpy as np
from cart_pole import q_learning

rng = np.random

class SDGregression:
    def __init__(self, D):
        # Model parameters
        w = np.random.rand(D) / np.sqrt(D)
        W = tf.Variable(w)
        # Model input and output
        self.x = tf.placeholder()
        self.y_hat = W * self.x
        self.y = tf.placeholder()
        # loss
        loss = tf.reduce_sum(tf.square(self.y_hat - self.y)) # sum of the squares
        # optimizer
        optimizer = tf.train.GradientDescentOptimizer(0.01)
        self.train = optimizer.minimize(loss)
        # training loop
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init) # reset values to wrong


    def update(self, X, Y):
        self.sess.run(self.train, {self.x:X, self.y:Y})

    def predict(self, X):
        return self.sess.run(self.y_hat, {self.x: X})

if __name__ == '__main__':
    q_learning.SGDRegressor = SDGregression
    q_learning.main()