import numpy as np
import pandas as pd
from random import randint
from Cluster.Distance import DistMinkov
from Cluster.Visulization import clusterClubs

def loadData(filename):
    '''
    读取数据
    :param filename: 文件名
    :return: 文件中的数据
    '''
    data = pd.read_csv(filename, sep=',', index_col=['number'])
    return data

class LVQ(object):
    def __init__(self, dist_parameter, lr=0.1, cluster_num=5):
        '''
        初始化 LVQ 类
        :param dist_parameter: 计算距离函数的相关信息
        :param lr: 学习率
        :param cluster_num: 聚类簇的数量
        '''
        self.lr = lr
        self.cluster_num = cluster_num
        self.nums = 0
        self.features = 0
        self.distant_func = dist_parameter['function']
        self.parameter = dist_parameter['parameter']
        self.cluster = {}

    def __initArgs__(self, Xdata, Ydata):
        '''
        初始化参数：学习向量
        :param Xdata: 特征值
        :param Ydata: 标签值
        :return: 初始化完成的特征向量
        '''
        self.nums, self.features = Xdata.shape
        # 选中的初始化学习向量
        index_arr = np.random.randint(1, self.nums, size=self.cluster_num)
        for key, index in enumerate(index_arr):
            self.cluster[key] = {
                'mean_vec': Xdata[index],
                'label': Ydata[index],
                'data': np.zeros((1, self.features))
            }

    def fit(self, Xdata, Ydata, iterations=100):
        '''
        拟合数据
        :param Xdata: 特征数据
        :param Ydata: 标签数据
        :param iterations: 迭代次数
        :return: 聚类结果
        '''
        self.__initArgs__(Xdata, Ydata)
        for iters in range(iterations):
            print('iteration:', iters)
            # 从样本空间中随机选一个作为训练数据
            train_index = randint(0, self.nums-1)
            Xtrain = Xdata[train_index]
            Ytrain = Ydata[train_index]

            min_dist = float(np.inf)
            update_index = -1
            # 计算该点到所有簇心的距离，取最短距离，记录学习向量的索引值
            for lv_index in self.cluster.keys():
                dist = self.distant_func(
                    Xtrain, self.cluster[lv_index]['mean_vec'],
                    p=self.parameter
                )
                if min_dist > dist:
                    min_dist = dist
                    update_index = lv_index
            # 更新参数
            self.update_learning_vector(
                Xtrain, Ytrain, update_index
            )

        return self.cluster


    def update_learning_vector(self, Xtrain, Ytrain, update_index):
        '''
        更新权重
        :param Xtrain: 训练数据的特征值
        :param Ytrain: 训练数据的标签值
        :param update_index: 需要更新的学习向量的索引
        :return:
        '''
        update_feature = self.cluster[update_index]['mean_vec']
        update_label = self.cluster[update_index]['label']

        if Ytrain == update_label:
            update_feature += self.lr * (Xtrain - update_feature)
        else:
            update_feature -= self.lr * (Xtrain - update_feature)
        self.cluster[update_index]['mean_vec'] = update_feature

    def predict(self, Xdata):
        '''
        对数据进行聚类
        :param Xdata: 数据特征值
        :return: 聚类结果
        '''
        for idata in Xdata:
            min_dist = float(np.inf)
            cluster_name = 0
            # 对于每个数据选出距离最近的簇
            for icluster in self.cluster.keys():
                dist = DistMinkov(
                    self.cluster[icluster]['mean_vec'],
                    idata,
                    p=2
                )
                if min_dist > dist:
                    min_dist = dist
                    cluster_name = icluster

            self.cluster[cluster_name]['data'] = np.r_[
                self.cluster[cluster_name]['data'],
                np.array([idata])
            ]
        return self.cluster


def main():
    filename = 'data/WaterMelon4.txt'
    data = loadData(filename)
    Xdata = data[['density', 'sugercontent']].values
    Ydata = data['label'].values
    dist_parameter = {
        'function': DistMinkov,
        'parameter': 2
    }
    clf = LVQ(
        dist_parameter,
        lr=0.1
    )
    clf.fit(
        Xdata,
        Ydata,
        400
    )
    cluster = clf.predict(Xdata)

    clusterClubs(cluster)
if __name__ == '__main__':
    main()