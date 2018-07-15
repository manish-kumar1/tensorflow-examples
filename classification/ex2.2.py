#!/usr/bin/python

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler

def plot_data(data):
  fig, axis = plt.subplots()
  m=['o', '+']
  c=['y', 'g']
  for i in range(len(data)):
    axis.scatter(data[i, 0], data[i, 1], marker=m[int(data[i, 2])], c=c[int(data[i, 2])]) 

  fig.show()
  c=input()

def ex2(file_path, alpha, epoch_range=50000, step=5000):
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

  #http://ronny.rest/blog/post_2017_09_11_tf_metrics/
  acc, acc_up = tf.metrics.accuracy(Y, tf.cast(tf.round(htx), tf.int32))

  optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(J)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
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
  ex2('ex2data2.txt', 0.001145)
