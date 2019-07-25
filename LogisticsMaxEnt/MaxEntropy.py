import numpy as np
import pandas as pd

def getData1():
    feature_names = ['年龄(特征1)', '有工作(特征2)', '有自己的房子(特征3)', '信贷情况(特征4)']
    label_names = ['类别(标签)']
    dataset = pd.read_csv('data/maxEntData2.csv', index_col='index')
    return dataset, feature_names, label_names


def getData2():
    dataset = pd.read_csv('data/maxEntData.csv', index_col='index')
    feature_names = ['FALSE', 'high', 'mild', 'overcast']
    label_names = ['y_label']
    return dataset, feature_names, label_names


class MaxEntropy(object):
    def __init__(self, EPS=0.005):
        # 经验分布 p~(x)
        self.p_x = {}
        # 经验分布 p~(x,y)
        self.p_xy = {}
        # 经验期望
        self._Ep_ = {}
        # 特征值(相当于二值特征函数fi)，编号：特征
        self.index_fi = {}
        # 特征值(相当于二值特征函数fi), 特征：编号
        self.fi_index = {}
        # 表示fi限制条件的个数
        self._n = 0
        # 权重w
        self.weights = {}
        self.last_weights = {}
        # 标签值
        self.label_values = set()
        # 特征数量
        self.feature_nums = 0
        # 数据量
        self.data_size = 0
        self.EPS = EPS

    def _getEp(self, index, train_x, train_y):
        # 计算第index个限制条件在数据集中出现的概率分布
        idx, fi_x, fi_y = self.index_fi[index]
        # idx, fi_x, fi_y = (2, 'hot', 'no')

        # print(self.index_fi)
        pyx = 0

        for X, Y in zip(train_x, train_y):
            if fi_x != X[idx]:
                continue
            else:
                pyx += self.predict(X)[fi_y] / self.data_size
        return pyx

    def _convergence(self):  # 判断是否全部收敛
        for key in self.fi_index.keys():
            if abs(self.last_weights[key] - self.weights[key]) >= self.EPS:
                return False
        return True

    def fit(self, iteration, train_x, train_y):
        self.initModel(train_x, train_y)
        for loop in range(iteration):
            print("iter:%d" % loop)
            self.last_weights = self.weights.copy()
            for i in range(self._n):
                ep = self._getEp(i, train_x, train_y)  # 计算第i个特征的模型期望
                self.weights[self.index_fi[i]] += np.log(self._Ep_[self.index_fi[i]] / ep) / self._n # 更新参数
            print("w:", self.weights)
            if self._convergence():  # 判断是否收敛
                print('break')
                break



    def initModel(self, train_x, train_y):
        '''
        初始化模型：
        1、计算经验分布
        2、初始化权重
        3、计算特征函数n的大小
        4、计算经验分布的期望
        :param train_x: 训练集特征值
        :param train_y: 训练集标签值
        :return:
        '''
        self.data_size, self.feature_nums = train_x.shape

        numXY = {}
        numX = {}

        # 遍历数据集中每一条数据
        for inum in range(self.data_size):
            # 遍历每一行，计算二值特征函数fi(x,y)
            for idx, values in enumerate(train_x[inum]):
                # idx代表特征的编号，values则为特征值，train_y[inum]为第inum条数据的类标签值
                xy = (idx, values, train_y[inum][0])

                self.label_values.add(train_y[inum][0])

                # 统计(x,y)联合检验分布的数量
                if xy not in numXY:
                    numXY[xy] = 0
                numXY[xy] += 1

                # 统计x的经验分布的数量
                if values not in numX:
                    numX[values] = 0
                numX[values] += 1

        # 计算联合经验概率，并给各个二值特征函数编号
        for index, xy in enumerate(numXY.keys()):
            self.p_xy[xy] = numXY[xy] / self.data_size
            self.index_fi[index] = xy
            self.fi_index[xy] = index
            self.weights[xy] = 0

        # 计算x的经验概率
        for _ in numX.keys():
            self.p_x[_] = numX[_] / self.data_size

        self._n = len(self.index_fi)

        # 由于fi是二值特征函数，所以E_p = P_xy
        self._Ep_ = self.p_xy

    def predict(self, feature):
        '''
        计算条件概率 P(y|x)
        :param x: 特征值
        :return: x属于各类标签的概率大小，返回形式为数组
        '''
        result = {}
        for ilabel in self.label_values:
            result[ilabel] = 0
            w_sum = 0
            for idx, x, y in self.fi_index.keys():
                # 判断fi为1的情况
                if y == ilabel and x == feature[idx]:
                    w_sum += self.weights[(idx, x, y)]
                else:
                    # 否则fi为0，跳过
                    continue
            result[ilabel] = np.exp(w_sum)

        # 除以正则化因子
        zw = sum(result.values())
        for ilabel in self.label_values:
            result[ilabel] /= zw
        return result


if __name__ == '__main__':
    dataset, feature_names, label_names = getData2()
    model = MaxEntropy()
    model.fit(1000, dataset[feature_names].values, dataset[label_names].values)