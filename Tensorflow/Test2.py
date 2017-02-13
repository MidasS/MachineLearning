import tensorflow as tf
import numpy as np

data =  np.array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
         [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
         [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
         [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]])

label = [[0, 0, 1, 0, 0],
         [1, 0, 0, 0, 0]]

input = 20
steps = 2
classes = 5
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
    feed_dict = {x: data, y:label}
    sess.run( init )

    for i in range(10000):
        sess.run(optimizer, feed_dict=feed_dict)
        if i % 100 == 0 :
            print (sess.run(accuracy, feed_dict))

    print (sess.run( pred, feed_dict={x: data}))