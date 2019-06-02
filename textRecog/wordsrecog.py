import numpy
#import matplotlib.pyplot
import scipy.special
import tensorflow as ts


"""
理解本段程序参考的信息来源：
一本名叫：Python 神经网路编程的书（Author：塔里克·拉希德 (Tariq Rashid)）
"""

"""
这个搭建的神经网络为最基础的2层神经网络=》 输入层，含隐层，输出层
"""
class neuralNetwork(object):


    """
    初始化函数，初始化输入节点，输出节点和含隐层节点
    再之后定义，输入层到隐藏层的权重和隐藏层到输出层的权重，并且赋予随机初始值
    最后定义学习率变量
    """
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.onodes = outputnodes
        self.hnodes = hiddennodes

        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        self.lr = learningrate
        """
        定义激活函数，激活函数主要解决了解决线性不可分的问题，不同激活函数的选取，激活函数种类很多，随着神经网络的优化，
        激活函数的适应性越来越高, 我这里主要来尝试传统sigmoid函数（书上提供的）
        """
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    """
     训练函数：处理训练数据，对整体网络的各级权重进行训练，寻找最优值
    """
    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)
        self.who += self.lr * numpy.dot(output_errors * final_outputs * (1 - final_outputs),
                                       numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot(hidden_errors * hidden_outputs * (1 - hidden_outputs), numpy.transpose(inputs))
        pass

    """
      执行函数：在网络训练后进行实际数据的检测
     """
    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs





