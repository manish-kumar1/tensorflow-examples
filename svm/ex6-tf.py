import tensorflow as tf
#import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
#from tkinter import *
import tensorflow.feature_column as fc
from tensorflow.contrib.learn.python.learn.estimators import linear

"""
def plot_data(raw_data):
    n = len(raw_data['y'])

    features = raw_data['X']
    labels = raw_data['y']

    fig, axis = plt.subplots()
    m = ['o', '+']
    c = ['y', 'b']

    for i in range(n):
        axis.scatter(features[i][0],
                     features[i][1],
                     marker=m[int(labels[i])], c=c[int(labels[i])])

    fig.show()
    input()

"""

def svm_input_fn(raw_data):
    features = tf.data.Dataset.from_tensor_slices(raw_data['X'])
    labels = tf.data.Dataset.from_tensor_slices(raw_data['y'])
    example_id = np.arange(len(raw_data['X'])).astype(str)
    dataset = tf.data.Dataset.zip((features, labels)).batch(1).shuffle(1024)
    x, y = dataset.make_one_shot_iterator().get_next()
    return {'x': x, 'example_id': example_id}, y


def cost1(v):
    # v >= 1: 0 ?
    if v >= np.ones(v.shape):
        return np.zeros(v.shape)
    return v

def cost0(v):
    if v <= -1*np.ones(v.shape):
        return np.zeros(v.shape)
    return v


class SVMClassifier(tf.estimator.Estimator):
    def __init__(self, optimizer):
        super(SVMClassifier, self).__init__(
            model_fn=linear.scda_model_fn,
            optimizer=optimizer
        )


def svm(raw_data):
    x = fc.numeric_column('x', shape=[1, 2])
    svm_optimizer = tf.contrib.linear_optimizer.SDCAOptimizer(example_id_column='x')
    #model = SVMClassifier(svm_optimizer)
    example_id = np.arange(len(raw_data['X'])).astype(str)
    input_fn_train = tf.estimator.inputs.numpy_input_fn(
        x={"example_id":example_id ,"x": raw_data['X']},
        y=raw_data['y'],
        num_epochs=None,
        shuffle=True)

    model = tf.contrib.learn.SVM(example_id_column='example_id', feature_columns=[x])
    model.fit(input_fn=input_fn_train, steps=1000)
    accuracy = model.evaluate(input_fn_train, step=1)
    print("accuracy: {}".format(accuracy))


if __name__ == '__main__':
    raw_data = loadmat('data/ex6data1.mat')
    svm(raw_data)


if __name__ == '__main__2':
    raw_data = loadmat('data/ex6data1.mat')

    train_x = raw_data['X']
    train_y = raw_data['y']

    X = tf.placeholder(dtype=tf.float32)
    Y = tf.placeholder(dtype=tf.float32)
    W = tf.get_variable(name='theta', shape=[1, 3])
    htx = tf.matmul(X, W, transpose_b=True)
    t1 = tf.matmul(Y, cost1(htx))
    t2 = tf.matmul(1-Y, cost0(htx))
    J = tf.constant(1)*tf.reduce_sum(t1+t2) + tf.reduce_sum(tf.square(W[1:]))/2.0

    acc, acc_up = tf.metrics.accuracy(Y, tf.cast(tf.round(htx), tf.int32))

    optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(J)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        #print("W={}".format(sess.run(W)))
        sess.run(W.assign([[1.0], [0.], [0.]]))
        for epoch in range(5000):
            _, c = sess.run([optimizer, J], feed_dict={X: train_x, Y: train_y})
            if epoch % 100 == 0:
                accuracy = sess.run([acc], feed_dict={X: train_x, Y: train_y})
                print("epoch: {}, cost={}, accuracy={}, W={}".format(epoch, c, accuracy, sess.run(W)))

            if (np.isnan(c)):
                break

        print("overall accuracy: {}".format(sess.run([acc], feed_dict={X: train_x, Y: train_y})))
