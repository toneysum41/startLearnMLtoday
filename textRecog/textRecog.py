import numpy as np
import struct
import os
import matplotlib.pyplot as plt
import wordsrecog

path = 'C:\\Users\\toney\\PycharmProjects\\tensorflowDemo\\MNIST_data'
weightIH = [[[0]*10 for _ in range(748)] for _ in range(100)]
weightHO = [[[0]*10 for _ in range(100)] for _ in range(10)]

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

def recogChar(num, network, test_images):
        temp = 0.0
        count = 0
        number = 0
        round = 1
        for i in test_images:
            result = network.query((np.asfarray(i) / 255) + 0.01)
            for r in result:
                if r - temp > 0.0:
                    temp = r
                    number = count
                count += 1
            print('Round', round, 'The number ', number, 'is found')
            round += 1
            count = 0
            temp = 0.0
            number = 0




def buildTrainingNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate):
    network = wordsrecog.neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    train_images, train_labels = load_mnist_train(path)
    #for j in range(10):
    for i in train_images:
            # img = i.reshape(28, 28)
            scaledinput_images = (np.asfarray(i) / 255.0 * 0.99) + 0.01
            onodes = 10
            targets = np.zeros(onodes) + 0.01
            targets[7] = 0.99
            network.train(scaledinput_images, targets)
       # weightIH[j] = network.wih
       # weightHO[j] = network.who
    test_images, test_labels = load_mnist_test(path)
    recogChar(9, network, test_images)


buildTrainingNetwork(784, 100, 10, 0.2)

#print(targets)

# fig, ax = plt.subplots(
#     nrows=10,
#     ncols=10,
#     sharex=True,
#     sharey=True, )
#
# ax = ax.flatten()
# count = 0
# for j in range(10):
#     for i in range(10):
#         img = train_images[train_labels == i][j].reshape(28, 28)
#         ax[count].imshow(img, cmap='Greys', interpolation='nearest')
#         count += 1
#     ax[j].set_xticks([])
#     ax[j].set_yticks([])
# plt.tight_layout()
# plt.show()




