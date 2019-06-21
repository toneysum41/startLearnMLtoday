from load_data import *
from wordsdivide import *


class WordsVector:
    def get_words_data(self):
        dataset = DataLoader().mealdataset
        worddivide = WordsDivide()
        worddivide.divide_train_data(dataset)


def run():
    data = WordsVector()
    data.get_words_data()


run()