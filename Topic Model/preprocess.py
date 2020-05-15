import numpy as np
import pandas as pd
from gensim.corpora.dictionary import Dictionary
import jieba

class Preprocess(object):
    def reader(self,path):
        self.data=pd.read_csv(path,encoding='utf-8')
        self.preprocess()
        self.createCorpus()

    def preprocess(self):
        # 加载停用词
        self.stopwords=set()
        with open("Topic Model/data/stopwords.txt",encoding='utf-8') as file:
            for line in file:
                self.stopwords.add(line[:-1])
        
        # 去除空行
        self.data.dropna(inplace=True)
        # 分词
        self.sentences=[]
        for line in self.data['content'].values.tolist():
            try:
                sentence=jieba.lcut(line)
                # 去除空格、数字、停用词    
                sentence=list(filter(lambda x: x.strip(),sentence))
                sentence=list(filter(lambda x: not str(x).isdigit(),sentence))
                sentence=list(filter(lambda x: x not in self.stopwords,sentence))
                self.sentences.append(sentence)
            except Exception:
                print('1')
                continue

    def createCorpus(self):
        self.dictionary=Dictionary(self.sentences)
        self.corpus=[self.dictionary.doc2bow(sentence) for sentence in self.sentences]