import numpy as np
import struct
import os
import matplotlib.pyplot as plt
import wordsrecog

path = 'C:\\Users\\toney\\PycharmProjects\\tensorflowDemo\\MNIST_data'
weightIH = [[[0]*10 for _ in range(748)] for _ in range(100)]
weightHO = [[[0]*10 for _ in range(100)] for _ in range(10)]

"""
分别读取训练数据和测试数据，两个文件都是来自于mnist提供的数据文件
"""
def load_mnist_train(path, kind='train'):
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    return images, labels


def load_mnist_test(path, kind='t10k'):
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    return images, labels

"""
我自己定义最后的执行识别的函数：将测试数据集循环放入神经网络的query中进行判定，每组结果中数值最大的判定为识别结果
最后再算一个总体识别概率
"""
def recogChar(num, network, test_images, test_labels):
        temp = 0.0
        count = 0
        number = 0
        round = 1
        accurate = 0
        for i in test_images:
            result = network.query((np.asfarray(i) / 255) + 0.01)
            for r in result:
                if r - temp > 0.0:
                    temp = r
                    number = count
                count += 1
            print('Round', round, 'The number ', number, 'is found', '(It should be', test_labels[round-1], ')')
            if number == test_labels[round-1]:
                accurate += 1
            round += 1
            count = 0
            temp = 0.0
            number = 0
        print('The accuracy is : ', (accurate/len(test_images))*100, '%')



"""
创建一个训练函数，使用训练数据集对你所创建的神经网络进行训练，这里是使用的10000次的训练数据
最终得到的就是所需要的带有优化过的权重值的神经网络，在这其中还可以尝试将训练后的权重结果进行
保存，使你训练的神经网络得以延续使用，例如在其他project中调用
"""
def buildTrainingNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate):
    network = wordsrecog.neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    train_images, train_labels = load_mnist_train(path)
    #for j in range(10):
    counter = 0
    for i in train_images:
            # img = i.reshape(28, 28)
            scaledinput_images = (np.asfarray(i) / 255.0 * 0.99) + 0.01
            onodes = 10
            targets = np.zeros(onodes) + 0.01
            targets[train_labels[counter]] = 0.99
            network.train(scaledinput_images, targets)
            counter += 1

       # weightIH[j] = network.wih
       # weightHO[j] = network.who
    test_images, test_labels = load_mnist_test(path)
    recogChar(9, network, test_images, test_labels)


buildTrainingNetwork(784, 100, 10, 0.2)






