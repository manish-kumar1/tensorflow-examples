#!/usr/bin/python

import numpy as np
import tensorflow as tf

def ex1(file_path, alpha, epoch_range=50000):
  raw_data = np.loadtxt(file_path, delimiter=',', dtype=float, unpack=True)
  
  #normalize data
  data = tf.keras.utils.normalize(raw_data)

  m = len(data[0])

  # reshape
  train_y = np.reshape(data[2], [m,1])
  train_x = np.reshape(np.column_stack((data[0],data[1])), [m,2])

  # build tf model
  with tf.device('/cpu:0'):
    X = tf.placeholder(dtype=tf.float32, shape=[m, 2])
    Y = tf.placeholder(dtype=tf.float32, shape=[m, 1])
    W = tf.get_variable("theta", shape=[2, 1], dtype=tf.float32)
    b = tf.Variable(m, dtype=tf.float32)

    pred = tf.matmul(X, W)
    J = tf.reduce_mean(tf.square(tf.subtract(pred,Y)))
    optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(J)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epoch_range):
      _, c = sess.run([optimizer, J], feed_dict={X: train_x, Y: train_y})
      if (epoch% 1000 == 0):
        print("epoch: {}, cost={}, W={}".format(epoch, c, sess.run(W)))

    # predict price
    x = [[2000.0, 3.0]]
    print("predict:{} = {}".format(x, sess.run(tf.matmul(x, W))))
    
if __name__ == '__main__':
  ex1('ex1data2.txt', 0.05)
