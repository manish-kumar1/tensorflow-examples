import tensorflow as tf
from scipy.io import loadmat
import numpy as np


batch_size=1
num_steps = 2000
learning_rate = 0.001


def cnn(x, n_classes, dropout, reuse, is_training):
    #
    with tf.variable_scope('conv_net', reuse=reuse):
        x = tf.reshape(x, shape=[-1, 20, 20, 1])

        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)
        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)
        
        fc1 = tf.contrib.layers.flatten(conv2)
        fc1 = tf.layers.dense(fc1, 1024)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        out = tf.layers.dense(fc1, n_classes)

        return out


def model_fn(features, labels, mode):
    logits_train = cnn(features, 10, 0.25, reuse=False, is_training=True)
    logits_test  = cnn(features, 10, 0.25, reuse=True, is_training=False)

    pred_classes = tf.argmax(logits_test, axis=1)
    pred_probas = tf.nn.softmax(logits_test)

    labels = tf.reshape(labels, [-1])

    print(pred_classes.shape)
    print(labels.shape)
    print(features.shape)
        # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op,
                                  global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})

    return estim_specs


if __name__ == '__main__':
    raw_data = loadmat('ex4data1.mat')
    weights = loadmat('ex4weights.mat')

    train_x = raw_data['X']
    train_y = raw_data['y']

    # update, since 0 is labeled as 10
    train_y[train_y == 10] = 0

    features_placeholder = tf.placeholder(train_x.dtype, train_x.shape)
    labels_placeholder = tf.placeholder(train_y.dtype, train_y.shape)

    features = tf.data.Dataset.from_tensor_slices(train_x)
    labels = tf.data.Dataset.from_tensor_slices(train_y).map(lambda y: y%10)

    dataset = tf.data.Dataset.zip((features, labels))
    iterator = dataset.make_initializable_iterator()

    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        sess.run(iterator.initializer, feed_dict={features_placeholder: train_x, labels_placeholder: train_y})

        #Build the Estimator
        model = tf.estimator.Estimator(model_fn)

        # Define the input function for training
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x=train_x, y=train_y,
            batch_size=batch_size, num_epochs=None, shuffle=True)
        # Train the Model
        model.train(iterator, steps=num_steps)

        # Evaluate the Model
        # Define the input function for evaluating
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x=raw_data['X'], y=raw_data['y'],
            batch_size=batch_size, shuffle=False)
        # Use the Estimator 'evaluate' method
        e = model.evaluate(input_fn)

        print("Testing Accuracy:", e['accuracy'])

