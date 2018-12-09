import numpy as np
import tensorflow as tf
from scipy.io import loadmat


def sigmoid_prime(z):
    return tf.multiply(z, 1-z)

def get_layer(X, theta, name):
  with tf.name_scope(name):
    W = tf.get_variable('theta_'+name, initializer=theta)
    return tf.sigmoid(tf.matmul(X, W, transpose_b=True))

def ex4(data_path, weights_path, lmda=1):
  raw_data = loadmat(data_path)
  weights = loadmat(weights_path)

  train_x = np.float32(raw_data['X']) # 5000x400
  train_y = np.int32(raw_data['y']) # 
  theta1 = np.float32(weights['Theta1']) # 25x401
  theta2 = np.float32(weights['Theta2']) # 10x26

  m, n = train_x.shape

  train_x = np.column_stack((np.ones(m), train_x))

  print(weights['Theta1'].shape)
  print(weights['Theta2'].shape)
  print(train_x.shape)
  print(train_y.shape)

  X = tf.placeholder(name='X', dtype=tf.float32)
  y = tf.placeholder(name='y', dtype=tf.float32)

  out1 = get_layer(X, theta1, 'hidden_layer1')
  out1 = tf.concat([tf.ones([m, 1], dtype=tf.float32), out1], 1)
  out2 = get_layer(out1, theta2, 'hidden_layer2')

  pred = tf.cast(tf.reshape(tf.argmax(out2, 1)+1, [-1, 1]), tf.float32)
  #pred2 = tf.reshape(tf.reduce_max(out2, 1), [-1, 1])

  t1 = tf.matmul(y, tf.log(out2), transpose_a=True)
  t2 = tf.matmul((1-y), tf.log(1-out2), transpose_a=True)

  J = (-1.0/m)*(tf.reduce_sum(t1+t2)) #non regularized
  theta_sum = tf.reduce_sum(tf.square(theta2[1:]))+tf.reduce_sum(tf.square(theta1[1:]))
  #REG_J = tf.add(J, (lmda/(2.0*m))*(tf.add(tf.reduce_sum(tf.square(theta2[1:])), tf.reduce_sum(tf.square(theta1[1:])))))
  REG_J = J + ((lmda/(2.0*m))*theta_sum)

  acc, acc_up = tf.metrics.accuracy(y, pred) 

  with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    cost, accuracy = sess.run([REG_J, acc_up], feed_dict={X:train_x, y: train_y})
    #a = sess.run([1-y], feed_dict={X:train_x, y: train_y})
    #b = sess.run([tf.log(1-pred2)], feed_dict={X:train_x, y:train_y})
    #print(a)
    #print(b)
    print(cost, accuracy) 

if __name__ == '__main__':
  ex4('ex4data1.mat', 'ex4weights.mat')
