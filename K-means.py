import copy

# 准备数据
file_path = 'D:\Download\machinelearninginaction\Ch10\\testSet.txt'


def laod_date(filepath):
    dataset = []
    labels = []
    file = open(filepath)
    for line in file.readlines():
        current_line = line.strip().split('\t')
        line_list = []
        for i in range(len(current_line[len(current_line) - 1])):
            line_list.append(float(copy.deepcopy(current_line[i])))
        dataset.append(line_list)
        labels.append(float(current_line[len(current_line) - 1]))
    return dataset, labels


def un_labels_data(filepath):
    dataset = []
    file = open(filepath)
    for line in file.readlines():
        current_line = line.strip().split('\t')
        line_list = []
        for i in range(len(current_line)):
            line_list.append(float(current_line[i]))
        dataset.append(line_list)
    return dataset


# 计算距离
def distince(dataset, martix):
    distince_martix = []
    for i in range(len(martix)):
        temp_i = []
        for j in range(len(dataset)):
            temp_j = 0
            for k in range(len(dataset[j])):
                temp_j += (dataset[j][k] - martix[i][k]) ** 2
            temp_i.append(temp_j ** 0.5)
        distince_martix.append(temp_i)
    return distince_martix


# 选取最近的簇
def choose_cluster(martix):
    result = []
    temp = []
    for k in range(len(martix)):
        temp.append(martix[k][0])
        result.append(0)
    for i in range(len(martix[0])):
        for j in range(len(martix)):
            if martix[j][i] < temp[j]:
                temp[j] = martix[j][i]
                result[j] = j
    return result


# 质心计算
def centroid(dataset, vector, n):
    temp = []
    for i in range(len(dataset[0])):
        temp_i = [0]*n
        for j in vector:
            temp_i[j] += dataset[i][j]
