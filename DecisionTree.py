from math import log
import copy


# 计算香农熵
def calculateEntropy(dataSet, labels):
    entropy = 0.0
    number = []
    for i in range(len(dataSet)):
        var = dataSet[i]
        if len(number) == 0:
            number.append([labels[var], 1])
        else:
            index = 0
            while index < len(number):
                if number[index][0] == labels[var]:
                    break
                index += 1
            if index == len(number):
                number.append([labels[var], 1])
            else:
                number[index][1] += 1

    for var in number:
        entropy -= var[1] / len(dataSet) * log(var[1] / len(dataSet), 2)
    return entropy


# 计算信息增益，并获取最大信息增益的feature，返回划分集合
def Gain(dataSet, labels):
    baseDataSet = list(range(len(labels)))
    baseEntropy = 0.0
    bestFeature = []
    baseEntropy = calculateEntropy(baseDataSet, labels)
    bestGain = 0.0
    numFeature = []
    temp_Gain = []
    feature = 0
    # 选取每 一种特征
    for i in range(len(dataSet[0])):
        numFeature.clear()
        for j in range(len(dataSet)):
            value = dataSet[j]
            if len(numFeature) == 0:
                numFeature.append([value[i], []])
                numFeature[0][1].append(j)
            else:
                index = 0
                while index < len(numFeature):
                    if value[i] == numFeature[index][0]:
                        break
                    else:
                        index += 1
                if index == len(numFeature):
                    numFeature.append([value[i], [j]])
                else:
                    numFeature[index][1].append(j)
        # temp_entropy = []
        temp_Gain = baseEntropy
        for k in range(len(numFeature)):
            # temp_entropy.append()
            temp_Gain -= len(numFeature[k][1]) / len(dataSet) * calculateEntropy(numFeature[k][1], labels)
        if temp_Gain > bestGain:
            bestGain = temp_Gain
            feature = i
        bestFeature.append(copy.deepcopy(numFeature))
    return feature, bestFeature[feature]


# 递归构建决策树结构，树结构用python内置的字典实现
Tree = {}


# 特征划分
def feature_division(mTree, dataSet, labels):
    temp_labels = []
    feature, numfeature = Gain(dataSet, labels)
    for k in range(len(dataSet)):
        del dataSet[k][feature]

    for i in range(len(numfeature)):
        temp_data = []
        temp_data.clear()
        temp_labels.clear()
        for var in numfeature[i][1]:
            temp_data.append(dataSet[var])
            temp_labels.append(labels[var])
        temp_item = 0
        for m in range(len(temp_labels)):
            if temp_labels[m] == temp_labels[0]:
                temp_item += 1
        if temp_item == len(temp_labels):
            mTree[numfeature[i][0]] = labels[numfeature[i][1][0]]
        else:
            if len(temp_data[0]) == 1:
                temp_sum = 0
                for n in range(len(temp_data)):
                    if temp_data[n] == temp_data[0]:
                        temp_sum += 1
                if temp_sum != len(temp_data):
                    mTree[numfeature[i][0]] = feature_division({}, temp_data, temp_labels)
            else:
                mTree[numfeature[i][0]] = feature_division({}, temp_data, temp_labels)
    DecisionTree = {}
    DecisionTree[feature] = mTree
    return DecisionTree


# 数据准备
# test_martix = [
# #     ['x', '0', '#'],
# #     ['x', '1', '@'],
# #     ['y', '1', '#'],
# #     ['x', '1', '#'],
# #     ['y', '0', '#'],
# #     ['x', '1', '@']
# # ]
# # test_labels = ['a', 'b', 'a', 'a', 'b', 'a']
# # Tree = feature_division({}, test_martix, test_labels)
# # print(Tree)
# 得到数据，将每个数据feature转化为向量，合并成martix,同时有一个labels向量
# 读取数据文件
filePath = 'D:\Download\machinelearninginaction\Ch03\lenses.txt'
# 决策树含有属性标签和类别标签
file = open(filePath)
lenses = []
for var in file.readlines():
    # print(var)
    temp_lenses= var.strip().split('\t')
    lenses.append(temp_lenses)
# print(lenses)
g_labels = []
for var in lenses:
    g_labels.append(copy.deepcopy(var[len(var) - 1]))
    del var[len(var) - 1]


# print(g_labels)
Tree = feature_division({}, lenses, g_labels)
print(Tree)

# 生成决策树


# 测试
