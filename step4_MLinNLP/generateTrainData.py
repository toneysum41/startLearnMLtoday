from convert2embedding import *

input_datas = []
for line in open('F:\\Users\\toney\\OneDrive\\Documents\\GitHub\\startLearnMLtoday\\step4_MLinNLP\\data\\fooddata.txt', encoding='gbk'):
    input_datas.append(line.split(' '))
convert2Embedding().convert_data(input_datas, modify_mode=True)