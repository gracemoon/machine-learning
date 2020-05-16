import numpy as np
import pandas as pd
from gensim.corpora.dictionary import Dictionary
import jieba
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer

class Preprocess(object):
    def reader(self,path):
        self.data=pd.read_csv(path,encoding='utf-8')
        self.data=shuffle(self.data)
        self.labels=self.data['label'].values
        self.countVectorizer=CountVectorizer()
        self.preprocess()
        self.createCorpus()

    def preprocess(self):
        # 加载停用词
        self.stopwords=set()
        with open("Classification/data/stopwords.txt",encoding='utf-8') as file:
            for line in file:
                self.stopwords.add(line[:-1])
        
        # 去除空行
        self.data.dropna(inplace=True)
        # 分词
        self.sentences=[]
        for line in self.data['segment'].values.tolist():
            try:
                sentence=jieba.lcut(line)
                # 去除空格、数字、停用词    
                sentence=list(filter(lambda x: x.strip(),sentence))
                sentence=list(filter(lambda x: not str(x).isdigit(),sentence))
                sentence=list(filter(lambda x: x not in self.stopwords,sentence))
                self.sentences.append(' '.join(sentence))
            except Exception:
                print('1')
                continue

    def createCorpus(self):
        self.sentences=self.countVectorizer.fit_transform(self.sentences).toarray()