from LogisticsMaxEnt.prepare import *
from collections import Counter
import random
import matplotlib.pyplot as plt

class Logistic(object):
    def __init__(self, input_vecs, labels, activation):
        '''
        感知机模型权重以及偏置项的初始化
        :param input_shape: <tuple>需要输入层维度信息
        :param activation: <funciton>激活函数
        :param lr: <float>学习率
        '''
        self.input_vecs = input_vecs.values
        self.labels = labels.values
        self.n_features = input_vecs.shape[1]
        self.n_nums = input_vecs.shape[0]
        self.activation = activation
        self.weight = np.ones((1, self.n_features))
        self.bias = np.ones((1, 1))

    def predict(self, input_vec):
        '''
        输入感知机运算结果
        :param input_vecs: <np.ndarray>训练数据
        :return:对ndarray中的逐元素应用激活函数
        '''
        output = np.dot(input_vec, self.weight.T).T
        return np.apply_along_axis(self.activation, axis=0, arr=output)


    #==============梯度上升优化算法=======================#
    def _batchGradientAscent(self, nums, lr):
        '''
        梯度上升优化算法
        ------
        :param nums: <np.ndarray>迭代次数
        :param lr: <np.ndarray>学习率
        :return:
        '''
        for k in range(nums):
            print('%d th iterations' % k)
            output = self.predict(self.input_vecs)
            delta = lr * (self.labels - output.T)
            delta_weight = np.dot(self.input_vecs.T, delta)
            self.weight += delta_weight.T

    # ==============随机梯度上升优化算法=======================#
    def _StochasticGradientAscent0(self, lr):
        '''
        随机梯度上升优化算法
        ------
        :param lr: <np.ndarray>学习率
        :return:
        '''
        for inum in range(self.n_nums):
            output = self.predict(self.input_vecs[inum])
            delta = lr * (self.labels[inum] - output.T)
            delta_weight = self.input_vecs[inum] * delta
            self.weight += delta_weight.T

    # ==============随机梯度上升优化算法=======================#
    def _StochasticGradientAscent1(self, nums):
        '''
        随机梯度上升优化算法
        ------
        :param nums: <np.ndarray>迭代次数
        :return:
        '''
        for iteration in range(nums):
            for inum in range(self.n_nums):
                data_index = [_ for _ in range(self.n_nums)]
                lr = 4 / (iteration + inum + 1) + 0.01
                rand_index = int(random.uniform(0, self.n_nums))
                output = self.predict(self.input_vecs[rand_index])
                delta = lr * (self.labels[rand_index] - output.T)
                delta_weight = self.input_vecs[rand_index] * delta
                self.weight += delta_weight.T
                del(data_index[rand_index])

    # ==============小批量梯度上升优化算法=======================#
    def _MiniBatchGradientAscent(self, nums, lr, batch_size=16):
        '''
        小批量梯度上升优化算法
        ------
        :param nums: <np.ndarray>迭代次数
        :param lr: <np.ndarray>学习率
        :param batch_size: <np.ndarray>批学习大小
        :return:
        '''
        for iteration in range(nums):
            for ibatch in range(1, self.n_nums // batch_size):
                start_index = (ibatch-1) * batch_size
                end_index = ibatch * batch_size

                mini_train_data = self.input_vecs[start_index: end_index, ]
                mini_label = self.labels[start_index: end_index, ]

                output = self.predict(mini_train_data)
                delta = lr * (mini_label - output.T)
                delta_weight = np.dot(mini_train_data.T, delta)
                self.weight += delta_weight.T

    def train(self, nums, optimization='gradAscent', lr=0.001):
        '''
        训练logistics模型
        :param nums: 迭代次数
        :param input_vecs: 训练样本的特征值
        :return:
        '''
        if optimization == 'gradAscent':
            self._batchGradientAscent(nums, lr)
        elif optimization == 'SGA0':
            self._StochasticGradientAscent0(lr)
        elif optimization == 'SGA1':
            self._StochasticGradientAscent1(nums)
        elif optimization == 'MGA':
            self._MiniBatchGradientAscent(nums, lr, batch_size=16)


    def classifyVector(self, inX):
        '''
        将计算出的概率转变为类别标签
        :param inX: 特征数据
        :return: 对应特征所属的类别
        '''
        prob = self.predict(inX)
        return np.apply_along_axis(
            lambda x: 1 if x > 0.5 else 0,
            axis=0,
            arr=prob
        )


    def accRate(self, y, y_hat):
        '''
        计算正确率
        :param y: 真实值
        :param y_hat: 模拟值
        :return: 正确率
        '''
        error = y - y_hat
        Counter(error)
        return sum(error == 0) / len(y)


def activation(x):
    '''
    对x中的所有元素逐元素的进行操作
    :param x:
    :return:
    '''
    return 1.0 / (1 + np.exp(-x))

def visualization(X):
    x_ponits = np.arange(4, 8)
    y_ = -(lrModel.weight[1] * x_ponits + lrModel.weight[0]) / lrModel.weight[2]
    plt.plot(x_ponits, y_)

    # lr_clf.show_graph()
    plt.scatter(X[:50, 0], X[:50, 1], label='0')
    plt.scatter(X[50:, 0], X[50:, 1], label='1')
    plt.legend()

if __name__ == '__main__':
    # train_data, feature_names, label_name = getData('data/testSet.txt')
    train_data, feature_names, label_name = createIrisData()
    lrModel = Logistic(
        train_data[feature_names],
        train_data[label_name],
        activation
    )
    lrModel.train(500, optimization='MGA')
    print(lrModel.weight)

