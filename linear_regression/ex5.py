
import numpy as np
import tensorflow as tf
from scipy.io import loadmat
#import matplotlib.pyplot as plt
#import tkinter


def display_data(data):
    x = data['X']
    y = data['y']
    plt.plot(x, y, 'r+')
    plt.ylabel('Water flowing out the dam (y)')
    plt.xlabel('Change in water level (x)')
    plt.axis([-50, 40, 0, 40])
    plt.show()
    

def cost_function(X, Y, W, lmda):
    m = X.shape[0].value
    with tf.name_scope('cost_fn'):
        htx = tf.matmul(X, W, transpose_b=True)
        error = tf.subtract(htx, Y)
        reg = (lmda)*tf.reduce_sum(tf.square(W[1:]))
        J = tf.reduce_sum(tf.square(error))
        return tf.add(J, reg)/(2.0*m)
        #return (tf.reduce_mean(tf.square(tf.subtract(tf.matmul(X, W), Y))) + (lmda/m)*tf.reduce_sum(tf.square(W[1:])))/2


def gradient_fn(X, Y, W, lmda):
    m = X.shape[0].value
    with tf.name_scope('gradient_fn'):
        # second term with first entry 0
        htx = tf.matmul(X, W, transpose_b=True) # 12x2, 2x1 = 12x1
        error = tf.subtract(htx, Y) # 12x1
        """
        array([[10, 10],
               [10, 20],
               [10, 30]], dtype=int32)
                >>> sess.run(tf.reduce_sum(z))
                    90
                >>> sess.run(tf.reduce_mean(z))
                    15
                >>> sess.run(tf.reduce_mean(z, axis=1))
                    array([10, 15, 20], dtype=int32)
                >>> sess.run(tf.reduce_mean(z, axis=0))
                    array([10, 20], dtype=int32)
        """
        term1 = tf.reduce_mean(tf.multiply(X, error), axis=0) # 12x2, 12x1 # axis = 0 i.e. column wise, = 1 i.e. row wise
        term2 = tf.divide(tf.multiply(W, lmda), m)
        # make zeroth column zero as j >= 1
        tmp = tf.constant([0, 1], dtype=tf.float64)
        term2 = tf.multiply(term2, tmp)
        return tf.add(term1, term2)


def polynomial_features(n_features):
    pass
    
def ex5(data_path='../../AndrewNg-ML/machine-learning-ex5/ex5/ex5data1.mat'):
    data = loadmat(data_path)

    #display_data(data)
    m, n = data['X'].shape

    train_x = np.column_stack([np.ones(m), data['X']])

    X = tf.placeholder(dtype=tf.float64, name='X', shape=train_x.shape)
    Y = tf.placeholder(dtype=tf.float64, name='Y', shape=data['y'].shape)
    W = tf.get_variable("theta", dtype=tf.float64, initializer=tf.fill([1, 2], np.float64(1)))

    lmbda = 1.0 #tf.constant(1.0, name='lmbda')
    alpha = 0.001

    J = cost_function(X, Y, W, lmbda)
    grad = gradient_fn(X, Y, W, lmbda)

    pred = tf.matmul(X, W, transpose_b=True)

    optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(J)

    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        cost = sess.run([J], feed_dict={X: train_x, Y: data['y']})
        theta = sess.run(W)
        print("theta = {}".format(theta))
        gradient = sess.run(grad, {X: train_x, Y: data['y']})
        print("cost = {}, gradient = {}".format(cost, gradient))

        for i in range(50000):
            _ = sess.run([optimizer],  {X: train_x, Y: data['y']})
            if i % 500 == 0:
                if np.isnan(cost):
                    break

                cost, theta= sess.run([J, W], {X: train_x, Y: data['y']})
                print("epoch: {}, cost = {}, W = {}".format(i, cost, theta))


if __name__ == '__main__':
    ex5()
