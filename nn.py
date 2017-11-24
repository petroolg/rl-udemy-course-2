# Implementation of a simple MLP network with one hidden layer. Tested on the iris data set.
# Requires: numpy, sklearn>=0.18.1, tensorflow>=1.0

# NOTE: In order to make the code simple, we rewrite x * W_1 + b_1 = x' * W_1'
# where x' = [x | 1] and W_1' is the matrix W_1 appended with a new row with elements b_1's.
# Similarly, for h * W_2 + b_2
import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from cart_pole import q_learning_rbf

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

class SDGregression:
    def __init__(self):
        # Layer's sizes
        x_size = 4 # Number of input nodes: 4 features and 1 bias
        h_size1 = 1000  # Number of hidden nodes
        h_size2 = 2000
        h_size3 = 50
        y_size = 1  # Number of outcomes (3 iris flowers)

        # Symbols
        self.X = tf.placeholder("float", shape=[None, x_size])
        self.Y = tf.placeholder("float", shape=[None, y_size])

        # Weight initializations
        w_1 = tf.Variable(tf.random_normal((x_size, h_size1), stddev=0.1))
        b_1 = tf.Variable(tf.random_normal((h_size1,), stddev=0.1))
        w_2 = tf.Variable(tf.random_normal((h_size1, h_size2), stddev=0.1))
        b_2 = tf.Variable(tf.random_normal((h_size2,), stddev=0.1))
        w_3 = tf.Variable(tf.random_normal((h_size2, h_size3), stddev=0.1))
        b_3 = tf.Variable(tf.random_normal((h_size3,), stddev=0.1))
        w_4 = tf.Variable(tf.random_normal((h_size3, y_size), stddev=0.1))
        b_4 = tf.Variable(tf.random_normal((y_size,), stddev=0.1))

        # Forward propagation
        self.yhat = forwardprop(self.X, w_1, b_1, w_2, b_2,w_3, b_3, w_4, b_4)

        # Backward propagation
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=self.yhat))
        self.updates = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

        # Run SGD
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def update(self, X, Y):
        self.sess.run(self.updates, {self.X:X, self.Y:Y})

    def predict(self, X):

        return self.sess.run(self.yhat, {self.X:X})



def forwardprop(X, w_1, b_1, w_2, b_2, w_3, b_3, w_4, b_4):
    """
    Forward-propagation.
    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
    """
    h = tf.nn.sigmoid(tf.matmul(X, w_1) + b_1)  # The \sigma function
    h = tf.nn.sigmoid(tf.matmul(h, w_2) + b_2)
    h = tf.nn.sigmoid(tf.matmul(h, w_3) + b_3)
    yhat = tf.matmul(h, w_4) + b_4  # The \varphi function
    return yhat

if __name__ == '__main__':
    q_learning_rbf.SGDRegressor = SDGregression
    q_learning_rbf.main()