import numpy as np

"""
softmax 分类函数及softmax 损失函数
softmax分类与logistic还有svm，knn等分类函数一样都是为了让目标对象更好的的被划分成两类或更多类，而softmax函数在划分类别
大于2时的问题会更有优势.所以作为处理神经网络的训练结果的分类函数，softmax很适合
"""
class softmax:
    def softmax(self, num_train, num_classes, y, W, dW):
        loss = 0.0
        for i in range(num_train):  # 在所有训练数据中按一层一层循环
            scores = X[i, :].dot(W)  # 含隐层的输出乘以权重矩阵获得输出层的结果
            scores_shift = scores - np.max(scores)  #防止数值溢出，将所有数值都减去结果中最大的数值
            right_class = y[i]  #当前训练的对象对应的正确结果
            loss += -right_class + np.log(np.sum(np.exp(scores_shift)))  #运用交叉熵算出softmax的损失
            for j in range(num_classes):   #遍历每一层的每个节点
                softmax_output = np.exp(scores_shift[j]) / np.sum(np.exp(scores_shift))  # 单个元素指数与所有元素指数和之比（softmax函数的公式）
                if j == y[i]:   # 如果当前节点属于正确的类别
                    dW[:, j] += (-1 + softmax_output) * X[i, :]   #调整含隐层到输出层权重
                else:   #如果不正确
                    dW[:, j] += softmax_output * X[i, :]
        return dW, loss
    """
    W: 权重， X：隐藏层输出数据， y：labels， reg：正则表达
    """
    def softmax_loss_p(self,W, X, y, reg):
        dW = np.zeros_like(W)

        num_train = X.shape[0]  #二维的长度
        num_classes = W.shape[1]  #一维的长度
        dW, loss = softmax(self, num_train, num_classes, y, W, dW)

        loss /= num_train
        loss += 0.5 * reg * np.sum()
        dW /= num_train
        dW += reg * W

        return loss, dW

