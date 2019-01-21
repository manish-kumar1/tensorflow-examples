import tensorflow as tf
import numpy as np
import tensorflow.feature_column as fc
from sklearn.preprocessing import MinMaxScaler


def input_fn(raw_data):
    dataset = tf.data.Dataset.from_tensor_slices(raw_data).map(
        lambda x: (x[0:2], x[-1])
    ).batch(batch_size=1).shuffle(len(raw_data))

    features, lables = dataset.make_one_shot_iterator().get_next()
    return {"x": features}, lables


def ex1(file_path):
    orig_data = np.loadtxt(file_path, delimiter=',')

    # scale data for feature normalization
    scaler = MinMaxScaler(copy=True, feature_range=(0,1))
    data = scaler.fit_transform(orig_data)

    # build model
    sqft = fc.numeric_column('sqft')
    bhk  = fc.numeric_column('bhk')
    price= fc.numeric_column('price')

    x = fc.numeric_column('x', shape=[1, 2])

    model = tf.estimator.LinearRegressor(feature_columns=[x],
                                         optimizer=tf.train.FtrlOptimizer(
                                            learning_rate=0.01,
                                            l1_regularization_strength=0.01))

    train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(data), max_steps=None)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn([[2000, 3, 30000]]), steps=1)

    # train the model
    #tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
    model.train(input_fn=lambda: input_fn(data))

    # test
    test_data = data.copy() #scaler.transform([[1650, 3, 0], [2000, 3, 0]])

    predict = model.predict(input_fn=lambda: input_fn(data))

    test_data[:, 2] = [p['predictions'] for p in predict]

    o = scaler.inverse_transform(test_data)
    for d in o:
        print("{} {} {}".format(*d))

    print(o-orig_data)

if __name__ == '__main__':
    ex1('ex1data2.txt')
