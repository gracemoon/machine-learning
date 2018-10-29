from math import exp

# 根据 machine learning in action ，采用梯度上升法求解

# 准备数据

filepath_test = 'D:\Download\machinelearninginaction\Ch05\\horseColicTest.txt'
filepath_train = 'D:\Download\machinelearninginaction\Ch05\\horseColicTraining.txt'
filepath_testset = 'D:\Download\machinelearninginaction\Ch05\\testSet.txt'


def classify(vector):
    result = []
    for i in range(len(vector)):
        if vector[i] >= 0.5:
            result.append(1)
        else:
            result.append(0)
    return result


def load_data(path):
    dataset = []
    labels = []
    file = open(path)
    for line in file.readlines():
        line_array = line.strip().split()
        temp_array = [1.0]
        for i in range(len(line_array) - 1):
            temp_array.append(float(line_array[i]))
        dataset.append(temp_array)
        labels.append(int(float(line_array[len(line_array) - 1])))
    return dataset, labels


# Sigmoid函数
def sigmoid(inX):
    a = exp(-inX)
    return 1.0 / (1 + a)


# 矩阵求导，求解梯度
def grad_scent(cycle_num):
    result = []

    dataset, labels = load_data(filepath_testset)
    weights = [0] * len(dataset[0])
    # 定义步长
    alpha = 0.001

    for i in range(cycle_num):
        direction = []
        temp_labels = []
        for j in range(len(labels)):
            inX = 0
            for k in range(len(weights)):
                inX += dataset[j][k] * weights[k]
            temp_labels.append(sigmoid(inX))
            direction.append(labels[j] - temp_labels[j])

            for m in range(len(weights)):
                temp_weights = 0
                for n in range(len(direction)):
                    temp_weights += dataset[n][m] * direction[n]
                weights[m] += alpha * temp_weights
        print(direction)
        print('labels:')
        class_labels = classify(labels)
        print(class_labels)
        class_temp_labels = classify(temp_labels)
        print('temp_labels:')
        print(class_temp_labels)
        error = 0
        for k in range(len(labels)):
            if class_temp_labels[k] != class_labels[k]:
                error += 1
        error_ratio = error / len(labels)
        print(error_ratio)
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>..')
    return weights


# 梯度上升法求解
def iteration(weight, martix, labels):
    result = []
    error = 0
    for i in range(len(martix)):
        result.append(0)
        for j in range(len(martix[i])):
            result[i] += weight[j] * martix[i][j]
    print('labels:')
    class_labels = classify(labels)
    print(class_labels)
    class_temp_labels = classify(result)
    print('temp_labels:')
    print(class_temp_labels)
    for k in range(len(labels)):
        if class_temp_labels[k] != class_labels[k]:
            error += 1
    error_ratio = error / len(labels)
    return error_ratio


# 绘制拟合曲线
def plot_best_fit(weights):
    import matplotlib.pyplot as plot
    dataset, labels = load_data(filepath_testset)
    xcord1 = []
    xcord2 = []
    ycord1 = []
    ycord2 = []
    for i in range(len(labels)):
        if labels[i] == 1:
            xcord1.append(dataset[i][1])
            ycord1.append(dataset[i][2])
        else:
            xcord2.append(dataset[i][1])
            ycord2.append(dataset[i][2])
    figure = plot.figure()
    ax = figure.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    temp = -3.0
    x = []
    y = []
    for i in range(60):
        temp += 1 * 0.1
        x.append(temp)
        y.append((-weights[0] - weights[1] * x[i]) / weights[2])
    ax.plot(x, y)
    plot.show()


# 测试
g_martix, g_labels = load_data(filepath_test)
plot_best_fit(grad_scent(500))

# print(iteration(grad_scent(500), g_martix, g_labels))
