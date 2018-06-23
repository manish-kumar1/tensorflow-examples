#!/usr/bin/python

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler

def plot_data(data):
  plt.figure(1)
  plt.subplot(211)
  plt.plot(data[2], data[0], 'ro')
  plt.subplot(212)
  plt.plot(data[2], data[1], 'ro')
  plt.show()

def ex1(file_path, alpha, epoch_range=100000, step=10000):
  raw_data = np.loadtxt(file_path, delimiter=',', dtype=int, unpack=True)
  m = len(raw_data[0])

  #plot_data(raw_data)

  #scale data
  data = np.column_stack((raw_data[0], raw_data[1]))
  scaler = MinMaxScaler(copy=False, feature_range=(0, 1))
  data = scaler.fit_transform(data)
  #data = tf.keras.utils.normalize(raw_data)

  # reshape
  train_y = np.reshape(raw_data[2], [m,1])
  train_x = np.reshape(np.column_stack((np.ones(m), data)), [m,3])

  # build tf model
  with tf.device('/cpu:0'):
    X = tf.placeholder(dtype=tf.float32, shape=[m, 3])
    Y = tf.placeholder(dtype=tf.float32, shape=[m, 1])
    W = tf.get_variable("theta", shape=[3, 1], dtype=tf.float32)
    b = tf.Variable(m, dtype=tf.float32)

    pred = tf.matmul(X, W)
    J = tf.reduce_mean(tf.square(tf.subtract(pred,Y)))
    optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(J)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epoch_range):
      _, c = sess.run([optimizer, J], feed_dict={X: train_x, Y: train_y})
      if (epoch % step == 0):
        print("epoch: {}, cost={}, W={}".format(epoch, c, sess.run(W)))

    # predict price on original data(don't do this at home ;)
    for i in range(m):
      x1 = [[raw_data[0][i], raw_data[1][i]]]
      y = [raw_data[2][i]]
      x = scaler.transform(x1)
      x = [[1.0, x[0][0], x[0][1]]]
      print("predict:{} = {}/{}".format(x1, sess.run(tf.matmul(x, W)), y))
    
if __name__ == '__main__':
  ex1('ex1data2.txt', 0.005)
