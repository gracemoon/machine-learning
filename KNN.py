from numpy import *
import operator
import matplotlib
from os import listdir
import matplotlib.pyplot as plt

g_figure = plt.figure()
ax = g_figure.add_subplot(111)

g_training_filePath = 'D:\Download\machinelearninginaction\Ch02\\trainingDigits'
g_test_filePath = 'D:\Download\machinelearninginaction\Ch02\\testDigits'


# 读取文件
def readFile(filename):
    in_vector = [0] * 1024
    file = open(filename)
    for i in range(32):
        lineStr = file.readline()
        for j in range(32):
            in_vector[32 * i + j] = int(lineStr[j])
    return in_vector


# 数据设置
def dataSet(dirPath):
    fileList = listdir(dirPath)
    martix = []
    labels = []

    for i in range(len(fileList)):
        filenameStr = fileList[i]
        fileStr = filenameStr.split('.')[0]
        labels.append(int(fileStr.split('_')[0]))
        martix.append(readFile(dirPath + '\\' + filenameStr))
    return martix, labels


# test_martix, test_labels = dataSet(g_training_filePath)
# # print(test_labels[1:5])
# print(test_martix[1:5])


# 向量距离计算
def distince(martixSet, vectorSet):
    Y = 0
    X_list = [0] * len(martixSet)
    XY = [0] * len(martixSet)
    for j in range(len((vectorSet))):
        Y += vectorSet[j] ** 2
    for i in range(len(martixSet)):
        for k in range(len(martixSet[i])):
            var = martixSet[i][k]
            X_list[i] += var ** 2
            XY[i] += var * vectorSet[k]
        X_list[i] += Y
        X_list[i] -= 2 * XY[i]
        X_list[i] = X_list[i] ** 0.5
    return X_list


# 结果选取
def select(listSet, k):
    selectedList = list(range(k))
    for i in range(k):
        selectedList[i] = 0
    for i in range(k):

        temp = listSet[0]
        tempIndex = 0
        for j in range(len(listSet)):
            var = listSet[j]
            if var < temp:
                temp = var
                tempIndex = j
        del listSet[tempIndex]
        selectedList[i] = tempIndex + i
    return selectedList


# classified
def classified(listSet, labels):
    array = list(range(len(listSet)))
    for k in range(len(listSet)):
        array[k] = 0
    for i in range(len(listSet)):
        for j in range(len(listSet)):
            if labels[listSet[i]] == labels[listSet[j]]:
                array[i] += 1
    temp = array[0]
    tempIndex = 0
    for m in range(len(listSet)):
        if temp < array[m]:
            temp = array[m]
            tempIndex = m
    return labels[listSet[tempIndex]]


# 数据可视化


# startup


# g_martixSet = [[4, 6], [7, 9], [9, 11], [7, 5], [2, 10],
#                [12, 8], [15, 10], [19, 8], [14, 8], [19, 10],
#                [15, 15], [11, 12], [16, 17], [13, 10], [15, 10],
#                [6, 10], [8, 14], [9, 12], [7, 13], [8, 18]]
# g_vectorSet = [10, 10]
# listSet = distince(g_martixSet, g_vectorSet)
# print(listSet)
# labels = ['a', 'a', 'a', 'a', 'a',
#           'b', 'b', 'b', 'b', 'b',
#           'c', 'c', 'c', 'c', 'c',
#           'd', 'd', 'd', 'd', 'd'
#           ]

training_martix, training_labels = dataSet(g_training_filePath)
test_martix, test_labels = dataSet(g_test_filePath)
g_k = 5
records = []
error = 0
# 循环识别每个测试数据
for i in range(len(test_martix)):
    listSet = distince(training_martix, test_martix[i])
    unselectedList = select(listSet, g_k)
    records.append(classified(unselectedList, training_labels))
    print('%d>>>>>>>>>>>>>>>>>>>>>>%d' % (records[i], test_labels[i]))

# 计算错误率
for j in range(len(records)):
    if records[j] != test_labels[j]:
        error += 1
error_ratio = error / len(test_labels)
print(error_ratio)
# print(unselectedList)
# print(classified(unselectedList, labels))

# 可视化
# position_X = [0] * len(g_martixSet)
# position_Y = [0] * len(g_martixSet)
# for index in range(len(g_martixSet)):
#     position_X[index] = g_martixSet[index][0]
#     position_Y[index] = g_martixSet[index][1]
# ax.scatter(position_X, position_Y)
# plt.show()
