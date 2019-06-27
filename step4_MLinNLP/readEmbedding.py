#coding:utf-8
import warnings
warnings.filterwarnings(action='ignore',category=UserWarning,module='gensim')
import re
from load_data import DataLoader
from itertools import islice
from gensim import *
from gensim.models.word2vec import Word2VecKeyedVectors
from fast2vec import fast2Vec

class EmbeddingData:
    def __init__(self):
        self.foodfilepath = 'F:\\Users\\toney\\OneDrive\\Documents\\GitHub\\startLearnMLtoday\\step4_MLinNLP\\data\\ChineseFoodEmbeddingModel.txt'
        self.txtfilepath = 'C:\\Users\\toney\\PycharmProjects\\embeddingdata\\ChineseEmbedding.txt'
        self.corpusfilepath = 'C:\\Users\\toney\\PycharmProjects\\embeddingdata\\basiccorpus.txt'
        self.regionfilepath = 'F:\\Users\\toney\\OneDrive\\Documents\\GitHub\\startLearnMLtoday\\step4_MLinNLP\\data\\regionname_clean.txt'

    def readcorpus(self):
        words = []
        for line in open(self.corpusfilepath, 'rb'):
            words.append(re.sub(re.compile(r'[^\u4e00-\u9fa5]'), "", line.decode('gbk')))
        return words

    def readregion(self):
        regions = []
        for line in open(self.regionfilepath, 'rb'):
            content = line.decode('gbk').split(',')
            regions.append(content[1].split(' '))
        return regions

    def filterEnbedding(self):
        vocab_dict_word2vec = {}
        dishnames = DataLoader().load_dish_name()
        basicwords = self.readcorpus()
        regionnames = self.readregion()
        print("arrive position1!")
        fv = fast2Vec()
        fv.load_word2vec_format(word2vec_path=self.txtfilepath)
        print("arrive position2!")
        for dishname in dishnames:
            basicwords +=dishname
        for regionname in regionnames:
            basicwords += regionname
        print("arrive position3!")
        for word in basicwords:
            vocab_dict_word2vec[word] = fv.wordVec(word, min_n=1, max_n=3)
        print("arrive position4!")
        fv.wordvec_save2txt(vocab_dict_word2vec, save_path='./data/ChineseFoodEmbeddingModel.txt', encoding='utf-8-sig')


    def read(self):
        embedding_text = Word2VecKeyedVectors.load_word2vec_format(self.foodfilepath, binary=False)
        return embedding_text

    def readorg(self):
        dic = {}
        dishnames = DataLoader().load_dish_name()
        basicwords = self.readcorpus()
        regionnames = self.readregion()
        embedding_text = Word2VecKeyedVectors.load_word2vec_format(self.txtfilepath, binary=False)
        model = embedding_text

        for word in basicwords:
            if word in model.wv.vocab.keys():
                dic[word] = model[word]
                a = 0
                temp =[]
                #print(dic[word])
                #print(dic[word][a])
                while a < 200:
                    temp.append(float(dic[word][a]))
                    a +=1
                dic[word] = temp

        for dishname in dishnames:
            for name in dishname:
                if name in model.wv.vocab.keys():
                    dic[name] = model[name]
                    a = 0
                    temp = []
                    while a < 200:
                        temp.append(float(dic[name][a]))
                        a +=1
                    dic[name] = temp
        for regionname in regionnames:
            for name in regionname:
                if name in model.wv.vocab.keys():
                    dic[name] = model[name]
                    a = 0
                    temp = []
                    while a < 200:
                        temp.append(float(dic[name][a]))
                        a +=1
                    dic[name] = temp
        f = open('./data/ChineseFoodEmbedding.txt', 'w', encoding='utf-8')
        index=0
        while index < 10:
            print(dic[basicwords[index]])
            index += 1
        for key in dic.keys():
            pattern = re.compile(r'[\[\]\n\r\t]')
            f.write(key+" "+re.sub(pattern, "", str(dic[key]))+'\n')
        f.close()

"""
    def read(self):
        f = open(self.txtfilepath, "rb")
        content = f.readlines()
        embeddingdatas = {}
        for data in content:
           datas = data.decode(encoding='utf-8').split(' ')
           label = datas[0]
           embedding = datas[1:]
           embeddingdatas[label] = embedding
        return embeddingdatas
"""
"""
This is used for reading .bin file
    def read(self):
        f = open(self.filepath, "rb")
        content = f.readlines()
        embeddingdatas = {}
        for data in content:
            datas = data.decode(encoding='gbk').split('\t')
            name = re.sub(r'b\'', "", datas[0])
            algenvalues = datas[1]
            embeddingdata = re.sub(r'\\r\\n', "", algenvalues)
            embeddingdatas[name] = embeddingdata
        return embeddingdatas
"""
"""
def test():
    data = EmbeddingData()
    #data.readorg()
    data.filterEnbedding()
test()
"""






