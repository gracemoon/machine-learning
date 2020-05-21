import gensim 
from gensim.models import LdaModel,TfidfModel
from preprocess import Preprocess
from sklearn.model_selection import train_test_split
import numpy as np 
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
import xgboost as xgb
from sklearn.metrics import accuracy_score

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

class XgboostModel(object):
    def __init__(self,model):
        self.model=model
        self.params = {  
        'booster': 'gbtree',     #使用gbtree
        'objective': 'multi:softmax',  # 多分类的问题、  
        # 'objective': 'multi:softprob',   # 多分类概率  
        #'objective': 'binary:logistic',  #二分类
        'eval_metric': 'merror',   #logloss
        'num_class': 4,  # 类别数，与 multisoftmax 并用  
        'gamma': 0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。  
        'max_depth': 8,  # 构建树的深度，越大越容易过拟合  
        'alpha': 0,   # L1正则化系数  
        'lambda': 10,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。  
        'subsample': 0.7,  # 随机采样训练样本  
        'colsample_bytree': 0.5,  # 生成树时进行的列采样  
        'min_child_weight': 3,  
        # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言  
        # 假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。  
        # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。  
        'silent': 0,  # 设置成1则没有运行信息输出，最好是设置为0.  
        'eta': 0.03,  # 如同学习率  
        'seed': 1000,  
        'nthread': -1,  # cpu 线程数  
        'missing': 1
        #'scale_pos_weight': (np.sum(y==0)/np.sum(y==1))  # 用来处理正负样本不均衡的问题,通常取：sum(negative cases) / sum(positive cases)  
    }  
        self.nums_round=5

    def dataPreprocess(self,path):
        self.preprocess=Preprocess()
        self.preprocess.reader(path)
        # 划分数据集
        self.train_data,self.test_data,self.train_labels,self.test_labels=train_test_split(self.preprocess.sentences,self.preprocess.labels
                                                                    ,test_size=0.3)
        print(self.train_labels[:100])
        self.xgb_train=xgb.DMatrix(np.array(self.train_data),label=np.array(self.train_labels))
        self.xgb_test=xgb.DMatrix(np.array(self.test_data))

    def train(self):
        self.bst=self.model.train(self.params,self.xgb_train,self.nums_round)
 
    def evaluation(self):
        output=self.bst.predict(self.xgb_test)
        print(accuracy_score(self.test_labels,output))

if __name__ == "__main__":
    # model=MultinomialNB()
    model=xgb
    # nb=NBModel(model)
    xgboostModel=XgboostModel(model)
    xgboostModel.dataPreprocess("Classification/data/police.csv")
    xgboostModel.train()
    xgboostModel.evaluation()



