import math
import copy

g_file_path = 'D:\Download\machinelearninginaction\Ch07\horseColicTraining2.txt'


# 准备数据
def load_data(file_path):
    dataset = []
    labels = []
    file = open(file_path)
    for line in file.readlines():
        current_line = line.strip().split('\t')
        labels.append(float(copy.deepcopy(current_line[len(current_line) - 1])))
        line_list = []
        for i in range(len(current_line) - 1):
            line_list.append(float(copy.deepcopy(current_line[i])))
        dataset.append(copy.deepcopy(line_list))
    return dataset, labels


# 获取最小值
def get_min(data):
    index = 0
    temp = data[0]
    for i in range(len(data)):
        if data[i] < temp:
            index = i
            temp = data[i]
    return data[index]


# 获取最大值
def get_max(data):
    index = 0
    temp = data[0]
    for i in range(len(data)):
        if data[i] > temp:
            index = i
            temp = data[i]
    return data[index]


# 矩阵取向量
def transfer(martix, index):
    result = []
    for i in range(len(martix)):
        result.append(martix[i][index])
    return result


# 构建及分类器
# 生成单层决策树(基于单个特征来做决策)
def simple_layer_decision(dataset, labals, weights):
    best_stump = []
    # 对于每个特征进行循环
    min_error_ratio = 1
    g_prediction_labels = []
    for i in range(len(dataset[0])):
        vector = transfer(dataset, i)
        range_min = float(get_min(vector))
        range_max = float(get_max(vector))
        step_size = (range_max - range_min) / 50
        for j in range(52):
            threshold = range_min + float(step_size * (j - 1))
            for item in ['lt', 'gt']:
                prediction_labels = judge(vector, threshold, item)
                error = [1] * len(prediction_labels)
                for k in range(len(prediction_labels)):
                    if prediction_labels[k] == labals[k]:
                        error[k] = 0
                error_ratio = 0
                for m in range(len(prediction_labels)):
                    error_ratio += error[m] * weights[m]
                temp_stump = [i, threshold, item]
                if min_error_ratio > error_ratio:
                    min_error_ratio = error_ratio
                    best_stump = temp_stump
                    g_prediction_labels = prediction_labels
    return best_stump, min_error_ratio, g_prediction_labels


# 根据阈值分类
def judge(data, threshold, flag):
    prediction = []
    if flag == 'lt':
        for i in range(len(data)):
            if data[i] <= threshold:
                prediction.append(1)
            else:
                prediction.append(-1)
    else:
        for i in range(len(data)):
            if data[i] <= threshold:
                prediction.append(-1)
            else:
                prediction.append(1)
    return prediction


# AdaBoost算法实现
def adaptive_boost(dataset, labels, cycle_num):
    # 初始化权重
    stump = []
    alpha = 0
    total_prediction_labels = [0] * len(dataset)
    weights = [1 / len(dataset)] * len(dataset)
    for i in range(cycle_num):
        # 获得基分类器、错误率、分类结果
        best_stump, min_error_ratio, g_prediction_labels = simple_layer_decision(dataset, labels, weights)
        # 计算基分类器系数alpha
        alpha = float(0.5 * math.log((1 - min_error_ratio) / min_error_ratio))
        best_stump.append(copy.deepcopy(alpha))
        # 将每次活得的基分类器加入分类器列表中
        stump.append(copy.deepcopy(best_stump))
        # 更新权重w
        total_weight = 0
        for j in range(len(weights)):
            total_weight += weights[j] * math.exp(-alpha * labels[j] * g_prediction_labels[j])
        for k in range(len(weights)):
            weights[k] = (weights[k] * math.exp(-alpha * labels[k] * g_prediction_labels[k])) / total_weight
        # 计算总误差
        total_error_ratio = 0
        total_error = 0
        for g in range(len(total_prediction_labels)):
            total_prediction_labels[g] += alpha * g_prediction_labels[g]
        sign_prediction = sign(total_prediction_labels)
        for f in range(len(sign_prediction)):
            if sign_prediction[f] != labels[f]:
                total_error += 1
        total_error_ratio = total_error / len(labels)
        print('the ' + str(i+1) + 'th total_error_ratio:' + str(total_error_ratio))
        if total_error_ratio == 0:
            break
    return stump


# 二分类函数
def sign(list):
    result = []
    for i in range(len(list)):
        if list[i] == 0:
            result.append(0)
        elif list[i] > 0:
            result.append(1)
        else:
            result.append(-1)
    return result


# 测试
def test():
    test_dataset, test_labels = load_data(g_file_path)
    adaptive_boost(test_dataset, test_labels, 50)


test()
