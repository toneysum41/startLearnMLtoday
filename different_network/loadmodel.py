import tensorflow as tf
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt
import numpy as np
import mnist

batch_size = 256

input_vec_size = 28
time_step_size = 28
mnist = mnist.read_data_sets('MNIST_data', one_hot=True)
with tf.Session() as session:
    get_model = tf.train.import_meta_graph('./model_results/LSTM_model.ckpt-157.meta')
    get_model.restore(session, tf.train.latest_checkpoint('./model_results/'))
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    batch_xs = batch_xs.reshape([batch_size, time_step_size, input_vec_size])
    print(session.run('weight_in:0'))

