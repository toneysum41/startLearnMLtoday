import re

class DataLoader:
    def __init__(self):
        self.datafile = 'data/regionname_clean.txt'
        self.dishdatafile = 'data/dishdata_clean.txt'
        self.traindatafile = 'data/traindata.txt'

    '''加载数据集'''
    def load_data(self):
        f = open(self.datafile, encoding='gbk')
        content = f.readlines()
        dataset = []
        for line in content:
            if line.strip('\n') is not'':
                line = line.strip().split(',')
                dataset.append([word for word in line[1].split(' ') if 'nbsp' not in word and len(word) < 11])
        return dataset

    def load_dish_name(self):
        dataset = []
        for line in open(self.dishdatafile, encoding='gbk'):
            line = line.strip().split(',')
            topword = line[1][:3]
            content = line[1][3:]
            dataset.append([word for word in content.split('  ')])
        return dataset

    def load_train_data(self, batch_size):
        dataset = {}
        labels = []
        for line in open(self.traindatafile, encoding='gbk'):
            line = line.strip()
            if re.search(re.compile(r'[^\u4e00-\u9fa5]'), line):
                line = line.split(',')
                for sentence in line:
                    dataset.append(sentence)
        return dataset, labels
"""
    def load_dish_data(self):
        dataset = []
        for line in open(self.dishdatafile, encoding='gbk'):
            line = line.strip()
            if re.search(re.compile(r'[^\u4e00-\u9fa5]'), line):
                line = line.split(',')
                for sentence in line:
                    dataset.append(sentence)
        return dataset
"""


names = DataLoader().load_dish_name()
print(names[0][0])