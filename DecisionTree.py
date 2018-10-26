from math import log


# 计算香农熵
def calculateEntropy(dateSet):
    entropy = 0.0
    for var in dateSet:
        entropy -= var * log(var, 2)
    return entropy


# 数据准备
# 生成决策树
