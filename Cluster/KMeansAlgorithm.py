import numpy as np
import pandas as pd
from Cluster.DistanceClass import DistMinkov
def load_data(filename):
    '''
    从文件中读取数据
    :param self:
    :param filename: 文件路径和文件名称
    :return: 返回读取的数据
    '''
    data = pd.read_csv(
        filename,
        sep='\t',
        header=None
    )
    return data

class KMeans(object):
    def __init__(self, cluster_num=5):
        '''
        初始化各簇的均值向量
        :param cluster_num: 簇的数量
        '''
        self.cluster_num = cluster_num
        self.nums = 0
        self.features = 0
        self.cluster = {}
        for i in range(self.cluster_num):
            self.cluster[i] = {
                'mean_vec': None,
                'data': None
            }

    def __initArgs__(self, data):
        '''
        随机选择cluster_num个样本值作为初始化的均值向量
        :param data: 数据集
        :return:
        '''
        self.nums, self.features = data.shape
        index = np.random.randint(1, self.nums, size=self.cluster_num)
        for key, value in enumerate(index):
            self.cluster[key]['mean_vec'] = data[value]
            self.cluster[key]['data'] = np.array([data[value]])

    def fit(self, data, iteration):
        # 1. 随机选取 cluster_num 个向量作为簇均值
        self.__initArgs__(data)
        for iters in range(iteration):
            # 2. 遍历样本空间,将每个样本进行归类
            for idata in data:
                min_dist = float(np.inf)
                cluster_name = 0
                # 选取一个最近的簇
                for icluster in self.cluster.keys():
                    dist = DistMinkov(self.cluster[icluster]['mean_vec'], idata, p=2)
                    if min_dist >= dist:
                        min_dist = dist
                        cluster_name = icluster
                # 将该条数据添加至最近的簇中
                self.cluster[cluster_name]['data'] = np.r_[
                    self.cluster[cluster_name]['data'],
                    np.array([idata])
                ]
            # 3. 更新簇均值向量
            stop = True
            for icluster in self.cluster.keys():
                # 计算新的簇心均值
                new_mean_vec = np.mean(
                    self.cluster[icluster]['data'], axis=0
                )
                # 重置数据
                self.cluster[icluster]['data'] = np.array([new_mean_vec])
                if ~(self.cluster[icluster]['mean_vec'] == new_mean_vec).all():
                    self.cluster[icluster]['mean_vec'] = new_mean_vec
                    stop = False
            if stop == True:
                break
        return self.cluster

if __name__ == '__main__':
    filename = 'data/testSet.txt'
    data = load_data(filename)
    clf = KMeans(4)

    cluster = clf.fit(data.values, 100)
    for i in cluster.keys():
        print(cluster[i]['mean_vec'])
