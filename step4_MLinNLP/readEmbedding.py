#coding:utf-8
import re
from gensim import *
from gensim.models.word2vec  import Word2VecKeyedVectors

class EmbeddingData:
    def __init__(self):
        self.binfilepath = './data/ChineseEmbedding.txt'
        self.txtfilepath = './data/ChineseEmbedding.txt'

    def read(self):
        embedding_text = Word2VecKeyedVectors.load_word2vec_format(self.txtfilepath, binary=False)
        model = embedding_text
        print(model.most_similar("如懿传"))
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

def test():
    data = EmbeddingData()
    data.read()
test()








