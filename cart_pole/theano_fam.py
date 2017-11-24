import gym
import theano
import numpy as np
from theano import tensor as T
from cart_pole import q_learning_rbf

rng = np.random


class SDGregression:
    def __init__(self, D):
        print('Hello theano!')
        w = np.random.rand(D) / np.sqrt(D)
        self.w = theano.shared(w)
        self.lr = 0.01

        X = T.matrix('X')
        Y = T.vector('Y')
        Y_hat = X.dot(self.w)
        delta = Y - Y_hat
        cost = delta.dot(delta)
        grad = T.grad(cost, self.w)
        updates = [(self.w, self.w - self.lr*grad)]
        self.train_op = theano.function(inputs=[X,Y], updates=updates)
        self.predict_op = theano.function(inputs=[X], outputs=Y_hat)

    def partial_fit(self, X, Y):
        self.train_op(X,Y)

    def predict(self, X):
        return self.predict_op(X)

if __name__ == '__main__':
    q_learning_rbf.SGDRegressor = SDGregression
    q_learning_rbf.main()