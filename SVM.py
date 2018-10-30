import random
import copy

# 支持向量机（SVM）support vector machines
# sequential minimal optimization(SMO)最小化序列算法
# 准备数据
g_filepath = 'D:\Download\machinelearninginaction\Ch06\\testSet.txt'


def set_data(filepath):
    dataset = []
    labels = []
    file = open(filepath)
    for line in file.readlines():
        line_array = line.strip().split('\t')
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


# print(select_max(1, 3))
# print(select_min(1, 3))


# SMO算法实现
def smo(dataset, label, C, toler, max_iter):
    b = 0.0
    alphas = [0.0] * len(dataset)
    iter = 0
    while iter < max_iter:
        print(iter)
        alpha_pair_changed = 0
        for i in range(len(dataset)):
            # temp_f = []
            # for g in range(len(dataset[0])):
            #     temp_f_sum = 0.0
            #     for h in range(len(dataset)):
            #         temp_f_sum += alphas[h] * label[h] * dataset[i][g]
            #     temp_f.append(temp_f_sum)
            # Fxi = 0.0
            # for l in range(len(temp_f)):
            #     Fxi += temp_f[l] * dataset[i][l]
            temp_f = []
            for g in range(len(dataset)):
                temp_f.append(alphas[g] * label[g])
            temp_martix = []
            for f in range(len(dataset)):
                temp_data = 0
                for o in range(len(dataset[0])):
                    temp_data += dataset[f][o] * dataset[i][o]
                temp_martix.append(copy.deepcopy(temp_data))
            Fxi = 0.0
            for l in range(len(temp_f)):
                Fxi += temp_f[l] * temp_martix[l]
            Fxi += b
            E1 = Fxi - label[i]
            if (label[i] * E1 < -toler and alphas[i] < C) or (label[i] * E1 > toler and alphas[i] > 0):
                j = select_param(i, len(dataset))
                if label[i] == label[j]:
                    L = select_max(0, alphas[j] + alphas[i] - C)
                    H = select_min(C, alphas[j] + alphas[i])
                else:
                    L = select_max(0, alphas[j] - alphas[i])
                    H = select_min(C, C + alphas[j] - alphas[i])
                if L == H:
                    # print("L==H")
                    continue
                K11 = 0.0
                K22 = 0.0
                K12 = 0.0
                for k in range(len(dataset[i])):
                    K11 += dataset[i][k] * dataset[i][k]
                    K22 += dataset[j][k] * dataset[j][k]
                    K12 += dataset[i][k] * dataset[j][k]
                K = K11 + K22 - 2 * K12
                if K <= 0:
                    # print("eta>=0")
                    continue
                temp_f_j = []
                for g in range(len(dataset)):
                    temp_f_j.append(alphas[g] * label[g])
                temp_martix_j = []
                for f in range(len(dataset)):
                    temp_data_j = 0
                    for o in range(len(dataset[0])):
                        temp_data_j += dataset[f][o] * dataset[j][o]
                    temp_martix_j.append(copy.deepcopy(temp_data_j))
                Fxj = 0.0
                for l in range(len(temp_f_j)):
                    Fxj += temp_f_j[l] * temp_martix_j[l]
                Fxj += b
                E2 = Fxj - label[j]
                E = E1 - E2
                new_alphas2 = alphas[j] + label[j] * E / K
                cliped_alpha2 = clip_alpha(new_alphas2, L, H)
                if abs(cliped_alpha2 - label[j]) < 0.0001:
                    # print("j not moving enough")
                    continue
                new_alphas1 = alphas[i] + label[i] * label[j] * (alphas[j] - cliped_alpha2)
                b1 = -E1 + (alphas[i] - new_alphas1) * label[i] * K11 + (alphas[j] - cliped_alpha2) * label[j] * K12 + b
                b2 = -E2 + (alphas[i] - new_alphas1) * label[i] * K12 + (alphas[j] - cliped_alpha2) * label[j] * K22 + b
                if 0 < new_alphas1 < C:
                    b = b1
                elif 0 < cliped_alpha2 < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2
                alphas[i] = copy.deepcopy(new_alphas1)
                # print(alphas[i])
                alphas[j] = copy.deepcopy(cliped_alpha2)
                alpha_pair_changed += 1
        if alpha_pair_changed == 0:
            iter += 1
        else:
            iter = 0
    return alphas, b


# 测试
def test():
    test_dataset, test_labels = set_data(g_filepath)
    alphas, b = smo(test_dataset, test_labels, 0.6, 0.001, 40)
    print(alphas)
    print(b)


test()
# print(clip_alpha(5, 1, 2))
# print(clip_alpha(0, 1, 2))
# print(select_param(1, 99))
