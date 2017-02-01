import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

n_input = 28
n_steps = 28
n_hidden = 128
n_classes = 10

batch_x, batch_y = mnist.train.next_batch(batch_size)

# test_label = mnist.test.labels[:test_len]
test_label = mnist.test.labels[:128]
# test_data = mnist.test.images[:128].reshape((-1, n_steps, n_input))
test_data = mnist.test.images[:128]

print(test_data.shape)
# print(batch_y)
# print(test_label)
print(test_label.shape)
# print(len(test_label))