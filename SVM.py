import random

# 支持向量机（SVM）support vector machines
# sequential minimal optimization(SMO)最小化序列算法
# 准备数据
g_filepath = 'D:\Download\machinelearninginaction\Ch06\\testSet.txt'


def set_data(filepath):
    dataset = []
    labels = []
    file = open(filepath)
    for line in file.readlines():
        line_array = line.strip().split('\\t')
        dataset.append([float(line_array[0]), float(line_array[1])])
        labels.append(float(line_array[2]))
    return dataset, labels

    # 修剪参数，再上下界之内


def clip_alpha(alpha, L, H):
    if alpha > H:
        alpha = H
    elif alpha < L:
        alpha = L
    return alpha


# 内选择，第二个alpha参数
def select_param(i, m):
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


# 选择最大数
def select_max(a, b):
    if a >= b:
        temp = a
    else:
        temp = b
    return temp


# 选择最小数
def select_min(a, b):
    if a < b:
        temp = a
    else:
        temp = b
    return temp


# SMO算法实现
def smo(dataset, label, C, toler, max_iter):
    b = 0
    alphas = [0] * len(dataset[0])
    iter = 0
    while iter < max_iter:
        for i in range(len(dataset[0])):
            j = select_param(i, len(dataset[0]))
            K11 = 0
            K22 = 0
            K12 = 0
            for k in range(len(dataset)):
                K11 += dataset[k][i] * dataset[k][i]
                K22 += dataset[k][j] * dataset[k][j]
                K12 += dataset[k][i] * dataset[k][j]
            K = K11 + K22 - 2 * K12
            E1 = alphas[i] * label[i] * K11 - label[i]
            E2 = alphas[j] * label[j] * K11 - label[j]
            E = E1 - E2
            new_alphas2 = alphas[j] + label[j] * E / K
            if label[i] == label[j]:
                L = select_max(0, alphas[j] + alphas[i] - C)
                H = select_min(C, alphas[j] + alphas[i])
            else:
                L = select_max(0, alphas[j] - alphas[i])
                H = select_min(C, C + alphas[j] - alphas[i])
            cliped_alpha2 = clip_alpha(new_alphas2, L, H)
            new_alphas1 = alphas[i] + label[i] * label[j] * (alphas[2] - cliped_alpha2)
            b1 = -E1 + (alphas[i] - new_alphas1) * label[i] * K11 + (alphas[j] - cliped_alpha2) * label[j] * K12 + b
            b2 = -E1 + (alphas[i] - new_alphas1) * label[i] * K12 + (alphas[j] - cliped_alpha2) * label[j] * K22 + b
            if 0 < new_alphas1 < C:
                b = b1
            elif 0 < cliped_alpha2 < C:
                b = b2
            else:
                b = (b1 + b2) / 2
            alphas[i] = new_alphas1
            alphas[j] = cliped_alpha2
        iter += 1

    return alphas, b


# 测试
def test():
    test_dataset, test_labels = set_data(g_filepath)
    smo(test_dataset, test_labels, 0.6, 0.001, 40)
