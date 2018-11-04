import copy
import random
import matplotlib.pyplot as plot

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
    for k in range(len(martix[0])):
        temp.append(martix[0][k])
    for m in range(len(martix[0])):
        result.append(0)
    for i in range(len(martix[0])):
        for j in range(len(martix)):
            if martix[j][i] < temp[i]:
                temp[i] = martix[j][i]
                result[i] = j
    return result


# 质心计算
def centroid(dataset, vector, n):
    temp = []
    for i in range(len(dataset[0])):
        temp_i = [0] * n
        sum = [0] * n
        for j in range(len(dataset)):
            temp_i[vector[j]] += dataset[j][i]
            sum[vector[j]] += 1
        for k in range(len(temp_i)):
            if sum[k] != 0:
                temp_i[k] /= sum[k]
            else:
                print(temp_i[k])
        temp.append(temp_i)
    result = []
    for l in range(len(temp[0])):
        temp_result = []
        for t in range(len(temp)):
            temp_result.append(temp[t][l])
        result.append(temp_result)
    return result


# 判断簇类别不再变化
def judge_cluster(first, second):
    result = 1
    for i in range(len(first)):
        if first[i] != second[i]:
            result = 0
            break
    return result


# 随机生成初始质心
def random_centroid(m, n):
    result = []
    while len(result) < m:
        flag = 0
        temp = int(random.uniform(0, n))
        if len(result) > 0:
            for i in range(len(result)):
                if temp == result[i]:
                    flag = 1
            if flag == 0:
                result.append(temp)
        else:
            result.append(temp)
    return result


# K-means算法实现
def k_means(dataset, num):
    centroid_init = random_centroid(num, len(dataset))
    init_nodes = []
    for item in centroid_init:
        init_nodes.append(dataset[item])
    over_flag = 1
    old_cluster = [-1] * len(dataset)
    while over_flag:
        distince_martix = distince(dataset, init_nodes)
        clusters = choose_cluster(distince_martix)
        judge_flag = judge_cluster(old_cluster, clusters)
        if judge_flag == 1:
            over_flag = 0
        else:
            old_cluster = clusters
            init_nodes = centroid(dataset, clusters, num)
    return init_nodes, clusters


# 绘制结果图
def plot_cord(dataset, clusters, n, centroids):
    cordinate = []
    for i in range(n):
        cordinate.append([])
    for j in range(len(clusters)):
        cordinate[clusters[j]].append(dataset[j])
    figure = plot.figure()
    ax = figure.add_subplot(111)
    cord_x1 = []
    cord_x2 = []
    cord_x3 = []
    cord_x4 = []
    cord_y1 = []
    cord_y2 = []
    cord_y3 = []
    cord_y4 = []
    for k in range(len(cordinate[0])):
        cord_x1.append(cordinate[0][k][0])
        cord_y1.append(cordinate[0][k][1])
    for k in range(len(cordinate[1])):
        cord_x2.append(cordinate[1][k][0])
        cord_y2.append(cordinate[1][k][1])
    for k in range(len(cordinate[2])):
        cord_x3.append(cordinate[2][k][0])
        cord_y3.append(cordinate[2][k][1])
    for k in range(len(cordinate[3])):
        cord_x4.append(cordinate[3][k][0])
        cord_y4.append(cordinate[3][k][1])
    centroid_x = []
    centroid_y = []
    for k in range(len(centroids)):
        centroid_x.append(centroids[k][0])
        centroid_y.append(centroids[k][1])
    ax.scatter(cord_x1, cord_y1, s=30, c='red')
    ax.scatter(cord_x2, cord_y2, s=30, c='green')
    ax.scatter(cord_x3, cord_y3, s=30, c='yellow')
    ax.scatter(cord_x4, cord_y4, s=30, c='blue')
    ax.scatter(centroid_x, centroid_y, s=30, c='black')
    plot.show()


def test():
    cluster_num = 4
    test_dataset = un_labels_data(file_path)
    # print(k_means(test_dataset, cluster_num))
    test_centroid, clusters = k_means(test_dataset, cluster_num)
    plot_cord(test_dataset, clusters, cluster_num, test_centroid)


test()
