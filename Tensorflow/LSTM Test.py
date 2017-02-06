# import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
# import numpy as np
#
#
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
#
# learning_rate = 0.001
# training_iters = 100000
# batch_size = 128
# display_step = 10
#
# n_input = 28
# n_steps = 28
# n_hidden = 128
# n_classes = 10
#
# batch_x, batch_y = mnist.train.next_batch(batch_size)
#
# # test_label = mnist.test.labels[:test_len]
# test_label = mnist.test.labels[:128]
# # test_data = mnist.test.images[:128].reshape((-1, n_steps, n_input))
# test_data = mnist.test.images[:128]
#
# print(test_data.shape)
# # print(batch_y)
# # print(test_label)
# print(test_label.shape)
# # print(len(test_label))


import tensorflow as tf
import numpy as np

from pandas import Series, DataFrame
import pandas as pd
import pandas.io.data as web
import datetime
import numpy as np
import tensorflow as tf
from sklearn import svm, preprocessing


class KOSPIDATA:

    def __init__(self):
        start = datetime.datetime(1998, 5, 1)
        end = datetime.datetime(2016, 12, 31)
        kospi = web.DataReader("^KS11", "yahoo", start, end)

        self.arr_date= np.array(kospi.index)
        self.arr_open = np.array(kospi.Open, dtype=float)
        self.arr_close= np.array(kospi['Adj Close'], dtype=float)
        self.arr_high= np.array(kospi.High, dtype=float)
        self.arr_low= np.array(kospi.Low, dtype=float)
        self.arr_volume= np.array(kospi.Volume, dtype=float)


if __name__ == "__main__":

    K = KOSPIDATA()

    # FEATURES = ['high', 'low', 'open', 'close', 'volume']
    FEATURES = ['high', 'low', 'open', 'close']

    data = {'year': K.arr_date,
            'open': K.arr_open,
            'high': K.arr_high,
            'low': K.arr_low,
            'close': K.arr_close}

    df = DataFrame(data, columns=['year', 'high', 'low', 'open', 'close'])
    # print(df[df.volume==0].index)

    # df = df.drop([ '484',  '823',  '828',  829,  830,  831,  832,  833,  956, 3887, 3927, 3945, 4097, 4101, 4105, 4109, 4112, 4154, 4155, 4205, 4427, 4614])
    # df = df.drop(df.ix[484], axis=1)
    # print(df.ix[484])


    def prediction():
        profit = []
        for i in range(len(df['close'])-1):
            if df['close'][i] < df['close'][i+1]:
                profit.append('1')
            else :
                profit.append('0')

        return profit

    profit = np.array(prediction())
    profit = np.append(profit,np.NaN)
    df['profit'] = profit



    print(df)

    new_XX = np.array(df[FEATURES].values[:-1, :])
    print(new_XX.shape)
    print(new_XX)
    new_YY = df['profit'].values[:-1]
    print(new_YY)

    #
    # data =  np.array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    #          [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
    #          [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
    #          [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]])
    #
    # label = [[0, 0, 1, 0, 0],
    #          [1, 0, 0, 0, 0]]

    input = 4
    steps = 2
    classes = 2
    batch_size = 2
    loss = 0.0
    hidden = 2


    x = tf.placeholder(tf.float32, shape=[None, input])
    y = tf.placeholder(tf.float32, shape=[None, classes])

    x_input = tf.split(0, 2, x)

    print(x_input)

    W = tf.Variable(tf.truncated_normal([hidden, classes]))
    b = tf.Variable(tf.truncated_normal([classes]))

    lstm = tf.nn.rnn_cell.BasicLSTMCell(steps)
    output, state = tf.nn.rnn(lstm, x_input, dtype=tf.float32 )

    pred = tf.nn.sigmoid(tf.matmul(output[-1], W ) + b)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

    correct_pred = tf.equal( tf.argmax(pred,1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        feed_dict = {x: new_XX, y:new_YY}
        sess.run(init)

        for i in range(10000):
            sess.run(optimizer, feed_dict=feed_dict)
            if i % 100 == 0 :
                print (sess.run(accuracy, feed_dict))

        print (sess.run( pred, feed_dict={x: new_XX}))
