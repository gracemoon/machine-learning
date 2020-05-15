import gensim 
from gensim.models import LdaModel,TfidfModel
from preprocess import Preprocess
from sklearn.model_selection import train_test_split
import numpy as np 
import pandas as pd


class TopicModel(object):
    def dataPreprocess(self,path):
        self.preprocess=Preprocess()
        self.preprocess.reader(path)

    def train(self):
        self.lda=LdaModel(self.preprocess.corpus,id2word=self.preprocess.dictionary,num_topics=10)
        for topic in self.lda.print_topics(num_topics=10,num_words=10):
            print(topic[1])

    def evaluation(self):
        pass

class TFIDFModel(object):
    def dataPreprocess(self,path):
        self.preprocess=Preprocess()
        self.preprocess.reader(path)

    def train(self):
        self.tfidf=TfidfModel(corpus=self.preprocess.corpus,id2word=self.preprocess.dictionary)
        data=self.tfidf[self.preprocess.corpus[0]]
        df=pd.DataFrame(np.array(data),columns=['index','tfidf'])
        df.sort_values(by='tfidf',ascending=False,inplace=True)
        id2token={}
        for key,value in self.preprocess.dictionary.token2id.items():
            id2token[value]=key
        print([id2token[x] for x in df['index'].values])

    def evaluation(self):
        pass


if __name__ == "__main__":
    # model=TopicModel()
    model=TFIDFModel()
    model.dataPreprocess("Topic Model/data/car.csv")
    model.train()


