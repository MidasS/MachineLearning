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



    #
    #
    # new_XX = np.array(df[FEATURES].values[:])
    # print(new_XX)
    #
    #
    #
    print(df)

    new_XX = np.array(df[FEATURES].values[1:-1, :])
    print(new_XX.shape)
    #
    # learning_rate = 0.001
    # training_iters = 100000
    # batch_size = 128
    # display_step = 10
    #
    # n_input = 4
    # n_steps = 20
    # n_hidden = 128
    # n_classes = 2
    #
    # x = tf.placeholder(tf.float32, [None, n_steps, n_input])
    # y = tf.placeholder(tf.float32, [None, n_classes])
    #
    # weights = tf.Variable(tf.random_normal([n_hidden, n_classes]))
    # biases = tf.Variable(tf.random_normal([n_classes]))
    #
    # x2 = tf.transpose(x, [1, 0, 2])
    # x2 = tf.reshape(x2, [-1, n_input])
    # x2 = tf.split(0, n_steps, x2)
    #
    # lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # outputs, states = tf.nn.rnn(lstm_cell, x2, dtype=tf.float32)
    # pred = tf.matmul(outputs[-1], weights) + biases
    #
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    # train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    #
    # correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    #
    # init = tf.initialize_all_variables()
    #
    # with tf.Session() as sess:
    #     sess.run(init)
    #     step = 1
    #
    #     while step * batch_size < training_iters:
    #         batch_x, batch_y = mnist.train.next_batch(batch_size)
    #         batch_x = batch_x.reshape((batch_size, n_steps, n_input))
    #
    #         sess.run(train, feed_dict={x: batch_x, y: batch_y})
    #         if step % display_step == 0:
    #             acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
    #             loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
    #             print("step : %d, acc: %f" % (step, acc))
    #         step += 1
    #     print("train complete!")
    #
    #     test_len = 128
    #     test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    #     test_label = mnist.test.labels[:test_len]
    #     print("test accuracy: ", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
