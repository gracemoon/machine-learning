from math import exp

# 根据 machine learning in action ，采用梯度上升法求解

# 准备数据

filepath = 'D:\Download\machinelearninginaction\Ch05\\testSet.txt'


def load_data(path):
    dataset = []
    labels = []
    file = open(path)
    for line in file.readlines():
        line_array = line.strip().split()
        dataset.append([float(line_array[0]), float(line_array[1])])
        labels.append(int(line_array[2]))
    return dataset, labels


# Sigmoid函数
def sigmoid(inX):
    return 1 / (1 + exp(-inX))


# 矩阵求导，求解梯度
def grad_scent(martix, cycle_num):
    result = []

    dataset, labels = load_data()
    weights = [1] * len(dataset[0])
    # 定义步长
    alpha = 0.001

    for i in range(cycle_num):
        direction = []
        temp_labels = []
        for j in range(len(labels)):
            inX = 0
            for k in range(len(labels)):
                inX += dataset[j][k] * weights[k]
            temp_labels.append(sigmoid(inX))
            direction.append(labels[j] - temp_labels[j])
            for m in range(len(weights)):
                temp_weights = 0
                for n in range(len(direction)):
                    temp_weights += martix[n][m] * direction[n]
                weights[m] += alpha * temp_weights

    return result

# 梯度上升法
