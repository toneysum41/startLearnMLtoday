from pyhanlp import *


class WordsDivide:
    def __init__(self):
        self.path = './data/fooddata.txt'

    def divide_train_data(self, traindata):
        NLPTokenizer = JClass("com.hankcs.hanlp.tokenizer.NLPTokenizer")
        HanLP.Config.ShowTermNature = False
        #CRFnewSegment = HanLP.newSegment("crf")
        f = open(self.path, 'w')
        for line in traindata:
            line = line.lstrip()
            print(line)
            wordList = NLPTokenizer.segment(line);
            for line in wordList:
                f.write(str(line) + " ")
            f.write('\n')
        f.close()




    def divide_input_data(self, text):
        wordList = HanLP.segment(text);
        for line in wordList:
            print(line)

