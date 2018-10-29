from math import exp

# 根据 machine learning in action ，采用梯度上升法求解

# 准备数据


def load_data():
    dataset = []
    labels = []
    return dataset, labels


# Sigmoid函数
def figmoid(inX):
    return 1 / (1 + exp(-inX))


# 矩阵求导，求解梯度
def grad_scent(martix, cycle_num):
    result = []
    temp_labels = []
    dataset, labels = load_data()
    # 定义步长
    alpha = 0.001

    for i in range(cycle_num):
        direction = []
        for j in range(len(labels)):
            direction.append(labels[i] - temp_labels[i])

    return result

# 梯度上升法
