import tensorflow as tf
#import matplotlib.pyplot as plt
from tensorflow.contrib import rnn

import numpy as np
import mnist
mnist = mnist.read_data_sets('MNIST_data', one_hot=True)

#configuration variable
learn_rate = 0.001
training_times = 10000

batch_size = 128

input_vec_size = 28
time_step_size = 28
n_hidden_units = 128
n_classes = 10

#tf graph input
x = tf.placeholder(tf.float32, [None, time_step_size, input_vec_size])
y = tf.placeholder(tf.float32, [None, n_classes])

weights = {
    'in': tf.Variable(tf.random_normal([input_vec_size, n_hidden_units])),
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}

biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units,])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes,]))
}

def RNN(X, weights, biases):
    X = tf.reshape(X, [-1, input_vec_size])
    X_in = tf.matmul(X, weights['in']) + biases['in']
    X_in = tf.reshape(X_in, [-1, time_step_size, n_hidden_units])

    cell = tf.contrib.rnn.GRUCell(n_hidden_units)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)

    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']

    return results

def Run():
    pred = RNN(x, weights, biases)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    train_op = tf.train.AdamOptimizer(learn_rate).minimize(cost)
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1)) #tf.atgmax对矩阵按行或列计算最大值
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) #先将correct_pred的数据类型用cast函数转化为float32,再用redcue_mean函数求平均值
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        step = 0
        while step * batch_size < training_times:
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            batch_xs = batch_xs.reshape([batch_size, time_step_size, input_vec_size])
            sess.run([train_op], feed_dict={
                x: batch_xs,
                y: batch_ys,
            })

            if step % 20 == 0:
                print(sess.run(accuracy, feed_dict={
                    x: batch_xs,
                    y: batch_ys,
                }))
            step += 1

Run()


