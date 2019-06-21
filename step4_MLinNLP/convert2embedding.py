from readEmbedding import *


class convert2Embedding:
    def __init__(self):
        self.filepath = './data/train_data.bin'

    def read_dictionary(self):
        embeddingdatas = EmbeddingData().read()
        return embeddingdatas

    def convert_data(self, input_data, modify_mode=False):
        embedding_data = []
        embedding_sentence = []
        labels = {'1': [0.97, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005], '2': [0.005, 0.97, 0.005, 0.005, 0.005, 0.005, 0.005]
                  , '3': [0.005, 0.005, 0.97, 0.005, 0.005, 0.005, 0.005], '4': [0.005, 0.005, 0.005, 0.97, 0.005, 0.005, 0.005]
                  , '5': [0.005, 0.005, 0.005, 0.005, 0.97, 0.005, 0.005], '6': [0.005, 0.005, 0.005, 0.005, 0.005, 0.97, 0.005]
                  , '7': [0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.97]}
        embedding_dic = self.read_dictionary()
        if modify_mode:
            for line in input_data:
                print(str(line))
                for word in line:
                    if word in embedding_dic.keys():
                        embedding_data.append(embedding_dic[word])
                    else:
                        print("The embedding dic can not find: "+word)
                print(
                    '请输入label数字(1.generalPOI_search,2.generalAdv_search,3.specific_search,4.POI_recommend,5.food_recommend,6,comparable_search,7.Introduce_dish)：\n')
                index = input()
                embedding_sentence = [labels[index],embedding_data]
                f = open(self.filepath, 'a')
                for i, w in enumerate(embedding_sentence[1]):
                    if i < len(embedding_sentence[1])-1:
                        f.write(w + '\n')
                    else:
                        f.write(w+',')
                f.write(',' + str(embedding_sentence[0]) + '\n')
            f.close()
        else:
            for line in input_data:
                for word in line:
                    embedding_data.append(embedding_dic[word])
                embedding_sentence = [embedding_data, None]
                f = open(self.filepath, 'a')
                for w in embedding_sentence[0]:
                    f.write(w + '\n')
                f.write(',' + "" + '\n')
            f.close()
        return embedding_sentence