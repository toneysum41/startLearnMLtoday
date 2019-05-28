import tensorflow as tf
#import matplotlib.pyplot as plt
from tensorflow.contrib import rnn

import numpy as np
import mnist
"""
理解本段程序参考过的article：
RNN原理： https://www.cnblogs.com/jiangxinyang/p/9362922.html
tf.contrib.rnn函数介绍： https://blog.csdn.net/MOU_IT/article/details/86103110
tf.nn.dynamic_rnn介绍： https://www.cnblogs.com/lovychen/p/9294624.html
tf.reduce_mean解释： https://blog.csdn.net/dcrmg/article/details/79797826
tf.unstack解释：https://blog.csdn.net/u012193416/article/details/77411535  

完全第一次接触神经网络可以想看下一些术语的解释：
http://www.dataguru.cn/article-12193-1.html
"""

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
x = tf.placeholder(tf.float32, [None, time_step_size, input_vec_size])  #创建一个三维占位符，分别根据LSTM网络的：输入向量，时间，和
y = tf.placeholder(tf.float32, [None, n_classes])

weights = {
    'in': tf.Variable(tf.random_normal([input_vec_size, n_hidden_units])),
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}

biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units,])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes,]))
}

"""
搭建一个LSTM或GRU RNN单一时刻网络
"""
def RNN(X, weights, biases):
    X = tf.reshape(X, [-1, input_vec_size])  # 将输入的向量重整形为x*定义的输入向量size的二维矩阵
    X_in = tf.matmul(X, weights['in']) + biases['in']  # 矩阵相乘，等同于a*b
    """
    将与输入权重矩阵相乘后的输入矩阵变量重整形为3维矩阵, 三个坐标分别为输入向量数量轴，时间轴，隐藏层节点数量轴
    因为RNN神经网络须有要满足对信息的记忆，所以神经网络的传递需要加入时间轴，用来记录信息再之前神经传递时的输出结果
    """
    X_in = tf.reshape(X_in, [-1, time_step_size, n_hidden_units])


    cell = tf.contrib.rnn.GRUCell(n_hidden_units)  #创建细胞状态，可以理解为信息传送带，在这个传送带上信息基本不会被改变
    init_state = cell.zero_state(batch_size, dtype=tf.float32)

    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']

    return results

def Run():
    pred = RNN(x, weights, biases)
    """
    tf.reduce_mean 函数用于计算张量tensor沿着指定的数轴上的的平均值，主要用作降维或者计算tensor的平均值。
    """
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


