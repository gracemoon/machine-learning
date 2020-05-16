import gensim 
from gensim.models import LdaModel,TfidfModel
from preprocess import Preprocess
from sklearn.model_selection import train_test_split
import numpy as np 
import pandas as pd
from sklearn.naive_bayes import MultinomialNB


class NBModel(object):
    def __init__(self,model):
        self.classifier=model

    def dataPreprocess(self,path):
        self.preprocess=Preprocess()
        self.preprocess.reader(path)
        # 划分数据集
        self.train_data,self.test_data,self.train_labels,self.test_labels=train_test_split(self.preprocess.sentences,self.preprocess.labels
                                                                    ,test_size=0.3)

    def train(self):
        self.classifier.fit(self.train_data,self.train_labels)
        print(self.classifier.score(self.train_data,self.train_labels))

    def evaluation(self):
        print(self.classifier.score(self.test_data,self.test_labels))



if __name__ == "__main__":
    model=MultinomialNB()
    nb=NBModel(model)
    nb.dataPreprocess("Classification/data/police.csv")
    nb.train()
    nb.evaluation()



