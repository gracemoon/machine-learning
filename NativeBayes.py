# 获取数据

# 公共函数
# 提取向量的所包含的属性值和数量（维数）


def extract_attribute(vector):
    martix = []
    for i in range(len(vector)):
        var = vector[i]
        if len(martix) == 0:
            martix.append([var, [i]])
        else:
            index = -1
            for j in range(len(martix)):
                if martix[j][0] == var:
                    index = j
            if index == -1:
                martix.append([var, [i]])
            else:
                martix[index][1].append(i)
    return martix

# 朴素贝叶斯算法实现
def native_bayes(martix, labels):
    result = extract_attribute(labels)
    temp_dataSet = []
    for i in range(len(result)):
        for j in range(len(martix[0])):
            temp_dataSet.clear()
            for k in result[i][1]:
                temp_dataSet.append(martix[k][j])
            result[i].append(extract_attribute(temp_dataSet))
    return result


# 分类算法实现


def classification(machine, dateset):
    total = 0
    best_classfication = []
    for e in range(len(machine)):
        total += machine[e][1]
    for i in range(len(dateset)):
        probability = [1] * len(machine)
        for k in range(len(probability)):
            probability[k] *= len(machine[k][1]) / total
            item = dateset[i]
            for j in range(len(item)):
                index = -1
                for g in range(len(machine[k][j + 2])):
                    if item[j] == machine[k][j + 2][g]:
                        index = g
                        break
                if index == len(machine[k][j + 2]):
                    probability[k] *= 1 / (len(machine[k][1]) + len(machine[k][j + 2]))
                else:
                    probability[k] *= (machine[k][j + 2][index] + 1) / (len(machine[k][1]) + len(machine[k][j + 2]))
        max_probability = probability[0]
        classify_num = 0
        for h in range(len(probability)):
            if probability[h] > max_probability:
                max_probability = probability[h]
                classify_num = h
        best_classfication.append(machine[classify_num][0])
    return best_classfication


# 数据处理


# 包含属性标签、属性值和类别标签


# 测试

import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
# 加载数据
from sklearn.datasets import load_digits
digits = load_digits()
print(digits.data.shape)

# 数据划分 
train_data,test_data,train_label,test_label=train_test_split(digits.data,digits.target,test_size=0.3)

# 创建模型
from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(train_data,train_label)


# 测试集准确率
acc=model.score(test_data,test_label)
print(acc)

# 类型预测
y_prob=model.predict_proba(test_data)
print(y_prob[:20])
print(np.argmax(y_prob,axis=1))
y_hat=model.predict(test_data)
print(y_hat)


from sklearn.metrics import confusion_matrix as cm

print(cm(test_label,y_hat))


