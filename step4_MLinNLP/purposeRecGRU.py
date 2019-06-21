import numpy as np
import tensorflow as tf

learn_rate = 0.001
training_times = 20000

batch_size = 128

input_vec_size = 100
time_step_size = 20
n_hidden_units = 128
n_classes = 7

# generalPOI_search(名词去搜索地区), generalAdv_search(形容词去搜索味道)，specific_search(名词去搜索菜名), POI_recommend(按地区去搜索菜名按评分排序),
# food_recommend(按食品名词去搜索菜名按评分排序), comparable_search(根据人名搜索他的喜好), Introduce_dish(名词搜索菜名)


weights = {
    'in': tf.Variable(tf.random_normal([input_vec_size, n_hidden_units]), name="weight_in"),
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]), name="weight_out")
}

biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ]), name="biases_in"),
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]), name="biases_out")
}


def RNN(X, weights, biases):
    X = tf.reshape(X, [-1, input_vec_size])
    X_in = tf.matmul(X, weights['in']) + biases['in']

    X_in = tf.reshape(X_in, [-1, time_step_size, n_hidden_units])

    cell = tf.contrib.rnn.GRUCell(n_hidden_units)  # 创建细胞状态，可以理解为信息传送带，在这个传送带上信息基本不会被改变

    init_state = cell.zero_state(batch_size, dtype=tf.float32)

    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']

    return results


def Run():
    # tf graph input
    x = tf.placeholder(tf.float32, [None, time_step_size, input_vec_size])  # 创建一个三维占位符，分别根据LSTM网络的：输入向量，时间，和
    y = tf.placeholder(tf.float32, [None, n_classes])
    pred = RNN(x, weights, biases)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))  # 作用为，可以加快模型收敛速度

    train_op = tf.train.AdamOptimizer(learn_rate).minimize(cost)
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))  # tf.atgmax对矩阵按行或列计算最大值
    accuracy = tf.reduce_mean(
        tf.cast(correct_pred, tf.float32))  # 先将correct_pred的数据类型用cast函数转化为float32,再用redcue_mean函数求平均值
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        step = 0
        preacc = 80
        while step * batch_size < training_times:
            #batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            batch_xs = batch_xs.reshape([batch_size, time_step_size, input_vec_size])
            train_result = sess.run(train_op, feed_dict={
                x: batch_xs,
                y: batch_ys,
            })

            if step % 20 == 0:
                print(train_result)
                """
                                for i in range(0, len(batch_xs)):
                    result = sess.run(correct_pred, feed_dict={x: batch_xs, y: batch_ys})
                    if not result[i]:
                        print('预测的值是：', sess.run(y, feed_dict={
                            x: batch_xs,
                            y: batch_ys,
                        }))
                        print('实际的值是：', sess.run(pred, feed_dict={
                            x: batch_xs,
                            y: batch_ys,
                        }))
                        one_pic_arr = np.reshape(batch_xs[i], (28, 28))
                        pic_matrix = np.matrix(one_pic_arr, dtype="float")
                        plt.imshow(pic_matrix)
                        plt.show()
                        #pylab.show()
                        break
                """
                acc = sess.run(accuracy, feed_dict={
                    x: batch_xs,
                    y: batch_ys,
                })
                print(acc)
                if acc * 100 > preacc:
                    saver.save(sess, './model_results/GRU_model.ckpt', global_step=step)
                    preacc = acc * 100
            step += 1



