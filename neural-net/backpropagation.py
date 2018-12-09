import numpy as np
import tensorflow as tf
from scipy.io import loadmat


def sigmoid_prime(z):
    return tf.multiply(z, (1-z))


def get_layer(X, theta, name):
  with tf.name_scope(name):
    # note feedfwd layer's matmul as was the case for classification
    return tf.sigmoid(tf.matmul(X, theta, transpose_b=True))


def back_propagation(thetaN, deltaN1, activationN, lmda, name='bplayer'):
    print(deltaN1.shape)
    print(thetaN.shape)
    print(activationN.shape)
    m = activationN.shape[0].value
    with tf.name_scope(name):
        p = tf.matmul(deltaN1, thetaN) # 5000x10 10x26   5000x26 
        deltaN = tf.multiply(p, sigmoid_prime(activationN), name = name+'delta') # 5000x26
        DeltaN = (1.0/m)*tf.matmul(deltaN1, activationN, transpose_a=True)

        reg = tf.concat([tf.zeros([thetaN.shape[0], 1]), (lmda/m)*thetaN[:, 1:]], axis=1)
        DeltaN += reg

        return deltaN, DeltaN


def cost_func(theta1, theta2, htx, y, lmda, m=5000):
    with tf.name_scope('cost_fn'):
        # pred2 = tf.reshape(tf.reduce_max(htx, 1), [-1, 1])
        # vectorize multiplication over K, note tf.multiply
        t1 = tf.multiply(y, tf.log(htx))
        t2 = tf.multiply((1 - y), tf.log(1 - htx))

        J = (-1.0 / m) * (tf.reduce_sum(t1 + t2))  # non regularized
        theta_sum = tf.reduce_sum(tf.square(theta2[1:])) + tf.reduce_sum(tf.square(theta1[1:]))
        # REG_J = tf.add(J, (lmda/(2.0*m))*(tf.add(tf.reduce_sum(tf.square(theta2[1:])), tf.reduce_sum(tf.square(theta1[1:])))))
        REG_J = tf.add(J, ((lmda / (2.0 * m)) * theta_sum))
        return REG_J


def ex4(data_path, weights_path, lmda=0.001):
    raw_data = loadmat(data_path)
    weights = loadmat(weights_path)

    train_x = np.float32(raw_data['X'])  # 5000x400
    train_y = np.float32(raw_data['y'])  # 5000x1
    # theta1 = np.float32(weights['Theta1']) # 25x401
    # theta2 = np.float32(weights['Theta2']) # 10x26

    theta1 = tf.get_variable('theta1',
                             dtype=tf.float32,
                             trainable=True,
                             initializer=tf.truncated_normal(weights['Theta1'].shape,
                                                             stddev=0.1,
                                                             seed=tf.set_random_seed(1234)))
    theta2 = tf.get_variable('theta2',
                             dtype=tf.float32,
                             trainable=True,
                             initializer=tf.truncated_normal(weights['Theta2'].shape,
                                                             stddev=0.1,
                                                             seed=tf.set_random_seed(1234)))

    m, n = train_x.shape

    train_x = np.column_stack((np.ones(m), train_x))

    # print(weights['Theta1'].shape) #25x401
    # print(weights['Theta2'].shape) #10x26
    # print(train_x.shape)
    # print(train_y.shape)

    a1 = X = tf.placeholder(name='X', dtype=tf.float32, shape=train_x.shape)
    y = tf.placeholder(name='y', dtype=tf.float32)

    a2 = get_layer(X, theta1, 'hidden_layer1')
    a2 = tf.concat([tf.ones([m, 1], dtype=tf.float32), a2], 1)  # 5000x26
    a3 = htx = get_layer(a2, theta2, 'hidden_layer2')  # 5000x10

    pred = tf.cast(tf.reshape(tf.argmax(a3, 1), [-1, 1]), tf.int32) # 5000x1

    J = cost_func(theta1, theta2, htx, y, lmda)

    acc, acc_up = tf.metrics.accuracy(tf.reshape(tf.argmax(y, 1), [-1, 1]), pred)

    dl3 = tf.subtract(a3, y)
    dl2, Dw2 = back_propagation(theta2, dl3, a2, lmda, name='bp2')
    dl1, Dw1 = back_propagation(theta1, dl2[:, 1:], a1, lmda, name='bp1')

    update = [theta1.assign(tf.subtract(theta1, Dw1)),
              theta2.assign(tf.subtract(theta2, Dw2))]

    y_ = tf.one_hot((train_y - 1), 10, axis=1)[:, :, 0]  #replaces with values at indices, with depth = 10

    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        #s1, s2 = sess.run([theta_sum, theta_sum2])
        #print(s1, s2)
        y_hot = sess.run(y_)
        for i in range(10000):
            u, _ = sess.run([update, acc_up], feed_dict={X: train_x, y: y_hot})
            if i % 100 == 0:
                cost, a, accuracy = sess.run([J, a3, acc], feed_dict={X: train_x, y: y_hot})
                if np.isnan(cost):
                    break

                print(cost, 100*accuracy)

        # a = sess.run([1-y], feed_dict={X:train_x, y: train_y})
        # b = sess.run([tf.log(1-pred2)], feed_dict={X:train_x, y:train_y})
        # print(a)
        # print(b)


if __name__ == '__main__':
    ex4('ex4data1.mat', 'ex4weights.mat')
