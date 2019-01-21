#!/usr/bin/python

import random
import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt
from scipy.io import loadmat

def display_digits(X):
  m, n = X.shape # 5000x400
  # randomly select 10 images
  for i in range(10):
    x = X[random.choice(range(m))]
    x = np.reshape(x, (20,20))
    plt.imshow(x)
    plt.show()

def sigmoidfn(z):
    return 1/(1+np.exp(-z))

def ex31(file_path):
  raw_data = loadmat(file_path)
  train_x = raw_data['X']
  train_y = raw_data['y']

  #display_digits(train_x)

  m, n = train_x.shape
  train_x = np.column_stack((np.ones(m), train_x))

  X = tf.placeholder(dtype=tf.float32, shape=train_x.shape, name="X")
  y = tf.placeholder(dtype=tf.float32, shape=train_y.shape, name="y")
  W = tf.get_variable('theta', shape=[n, 1], dtype=tf.float32)

  

def ex3(file_path, alpha=0.001, epoch_range=50000, step=5000):
  raw_data = np.loadtxt(file_path, delimiter=',', unpack=False)
  plot_data(raw_data)

  train_x = np.column_stack((np.ones(len(raw_data)), raw_data[:, 0:2]))
  train_y = np.asmatrix(raw_data[:, 2]).T

  m, n = train_x.shape

  X = tf.placeholder(dtype=tf.float32, shape=train_x.shape, name="X")
  Y = tf.placeholder(dtype=tf.float32, shape=train_y.shape, name="Y")
  W = tf.get_variable("theta", shape=[3, 1], dtype=tf.float32)

  htx = tf.sigmoid(tf.matmul(X, W))
  t1 = tf.matmul(Y, tf.log(htx), transpose_a=True)
  t2 = tf.matmul((1-Y), tf.log(1-htx), transpose_a=True)
  
  J = tf.reduce_sum(tf.square(W[1:]))/(2.0 * m) + (-1.0/m)*tf.reduce_sum(t1+t2)

  acc, acc_up = tf.metrics.accuracy(Y, tf.cast(tf.round(htx), tf.int32))

  optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(J)

  with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    #print("W={}".format(sess.run(W)))
    sess.run(W.assign([[1.0], [0.], [0.]]))
    for epoch in range(epoch_range):
      _, c, _  = sess.run([optimizer, J, acc_up], feed_dict={X: train_x, Y: train_y})
      if (epoch % step == 0):
        accuracy = sess.run([acc], feed_dict={X: train_x, Y: train_y})
        print("epoch: {}, cost={}, accuracy={}, W={}".format(epoch, c, accuracy, sess.run(W)))

      if (np.isnan(c)):
        break

    print("overall accuracy: {}".format(sess.run([acc], feed_dict={X: train_x, Y: train_y})))
  
    
if __name__ == '__main__':
  ex3('ex3data1.mat')
  print(raw_data)
