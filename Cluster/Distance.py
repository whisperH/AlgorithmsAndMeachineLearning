import numpy as np


def DistMinkov(xi, xj, p=2):
    '''
    计算两个向量的曼哈顿距离(第一范数)
    :param xi: 向量xi, <1*n>
    :param xj: 向量xj, <1*n>
    :return: xi与xj之间的曼哈顿距离
    '''
    x = xi - xj
    data = np.apply_along_axis(np.power, 0, x, p)
    sum_data = np.sum(data)
    return pow(sum_data, 1/p)

def DistVDM(xi, xj):
    pass


