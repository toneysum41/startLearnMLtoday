import tensorflow as tf
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt
import numpy as np
import mnist

batch_size = 1

input_vec_size = 28
time_step_size = 28
n_hidden_units = 128
n_classes = 10
mnist = mnist.read_data_sets('MNIST_data', one_hot=True)

x = tf.placeholder(tf.float32, [None, time_step_size, input_vec_size])  #创建一个三维占位符，分别根据LSTM网络的：输入向量，时间，和
y = tf.placeholder(tf.float32, [None, n_classes])


def RNN(X, weights, biases):
    X = tf.reshape(X, [-1, input_vec_size])  # 将输入的向量重整形为x*定义的输入向量size的二维矩阵
    X_in = tf.matmul(X, weights['in']) + biases['in']  # 矩阵相乘，等同于a*b
    X_in = tf.reshape(X_in, [-1, time_step_size, n_hidden_units])
    cell = tf.contrib.rnn.GRUCell(n_hidden_units)  #创建细胞状态，可以理解为信息传送带，在这个传送带上信息基本不会被改变
    init_state = cell.zero_state(batch_size, dtype=tf.float32)

    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']

    return results


with tf.Session() as session:
    get_model = tf.train.import_meta_graph('./model_results/LSTM_model.ckpt-440.meta')
    get_model.restore(session, tf.train.latest_checkpoint('./model_results/'))
    graph = tf.get_default_graph()
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    batch_xs = batch_xs.reshape([batch_size, time_step_size, input_vec_size])
    weights = {
        'in': session.run('weight_in:0'),
        'out': session.run('weight_out:0')
    }

    biases = {
        'in': session.run('biases_in:0'),
        'out': session.run('biases_out:0')
    }
    #print('读入ih权重： ', weights['in'])
    #print('读入ho权重： ', weights['out'])
    pred = RNN(x, weights, biases)
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    init = tf.global_variables_initializer()
    session.run(init)
    pred_result = session.run(pred, feed_dict={
        x: batch_xs,
        y: batch_ys
    })
    correct_pred_result = session.run(correct_pred, feed_dict={
        x: batch_xs,
        y: batch_ys
    })
    print("预测概率： ", pred_result)
    print("预测值：", np.argmax(pred_result))
    print("预测结果： ", correct_pred_result)
    print("读取值： ", batch_ys)
    one_pic_arr = np.reshape(batch_xs, (28, 28))
    pic_matrix = np.matrix(one_pic_arr, dtype="float")
    plt.imshow(pic_matrix)
    plt.show()


