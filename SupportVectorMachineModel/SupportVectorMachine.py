import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def create_data():
    '''
    准备数据
    :return: 特征值和标签值
    '''
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    print(df.shape)
    df['label'] = iris.target
    df.columns = [
        'sepal length', 'sepal width', 'petal length', 'petal width', 'label'
    ]
    # 取前100条数据，2类。
    data = np.array(df.iloc[:100, [0, 1, -1]])
    for i in range(len(data)):
        if data[i, -1] == 0:
            data[i, -1] = -1
    # print(data)
    return data[:, :2], data[:, -1]

class SVM(object):
    def __init__(self, iterations, margin=0, penalty=1, kernal='linear'):
        '''
        类初始化
        :param iterations: 迭代次数
        :param margin: 松弛变量
        :param penalty: 惩罚参数，可用于限制alpha的范围
        :param kernal: 核方法，默认使用线性，可选的有多项式
        '''
        self.kernal = kernal
        self.iterations = iterations
        self.margin = margin
        self.penalty = penalty

    def __initArgs__(self, _X, _Y):
        '''
        根据数据集初始化alpha等重要参数
        :param _X: 训练集中的特征值
        :param _Y: 训练集中的标签值
        :return:
        '''
        self.trainX = _X
        self.trainY = _Y
        print('train_X shape:', self.trainX.shape)
        print('train_Y shape:', self.trainY.shape)
        # 表示数据量大小
        self.data_size = _X.shape[0]
        # 表示特征的数据量
        self.feature_nums = _X.shape[1]
        self.alpha = np.ones(_Y.shape)
        self.b = np.zeros((1, 1))
        # 预测值与真是输出之差
        self._E = [self._getE(_) for _ in range(self.data_size)]

    def fit(self, x, y):
        '''
        通过x, y拟合模型
        :param x: 训练集中的特征值
        :param y: 训练集中的标签值
        :return:
        '''
        self.__initArgs__(x, y)
        self.SMO()
        print('train done.')

    def kernalFunction(self, x1, x2):
        '''
        核函数计算结果：
        linear：线性函数，x1与x2的内积
        :param x1: 特征值1
        :param x2: 特征值2
        :param parameter: 核函数种类：线性和多项式
        :return: 映射结果
        '''
        if self.kernal == 'linear':
            return np.dot(x1, x2.T)
        elif self.kernal == 'poly':
            return np.dot(x1, x2.T)**2
        return 0

    def _g(self, xi):
        '''
        计算 wx+b 的结果
        :param xi: 1*d 维的特征值
        '''
        # w维度为(N*1)
        w = np.multiply(self.alpha, self.trainY)
        # kernal_result维度为(1*N)
        kernal_result = self.kernalFunction(xi, self.trainX)
        res = np.dot(kernal_result, w) + self.b
        return res

    def checkKKT(self, index_no):
        '''
        判断第index_no行数据(即alpha[index_no])是够符合KKT条件：
        :param index_no: 下标
        :return:
        '''

        xi = self.trainX[index_no, :]
        gx = self._g(xi)
        if self.alpha[index_no] == 0:
            return self.trainY[index_no] * gx > 1 - self.margin
        elif self.penalty > self.alpha[index_no] > 0:
            return self.trainY[index_no] * gx == 1 - self.margin
        else:
            return self.trainY[index_no] * gx < 1 - self.margin

    def _getE(self, index_no):
        '''
        计算误差E
        :param index_no: 目标数据的编号
        :return:
        '''
        xi = self.trainX[index_no, :]
        return self._g(xi) - self.trainY[index_no]

    def accurate(self, X, Y):
        '''
        计算准确度
        :param X: 维度是(M*d)的特征值
        :return:
        '''
        right_num = 0
        for i in range(X.shape[0]):
            if Y[i] == self.predict(X[i]):
                right_num += 1
        return right_num / X.shape[0]

    def predict(self, x):
        '''
        预测
        :param x: 1*d 维的特征值
        :return: 预测结果
        '''
        return 1 if self._g(x) > 0 else -1

    def init_alpha(self):
        '''
        选择alpha1和alpha2变量，其中，alpha1是违背KKT条件的变量，
        alpha2是与alpha1对应偏差最大的下标
        :return: alpha1和alpha2的下标
        '''
        # 优先检查可能是支持向量的alpha
        satisfy_index = []
        unsatisfy_index = []
        for i in range(self.data_size):
            if 0 < self.alpha[i] < self.penalty:
                satisfy_index.append(i)
            else:
                unsatisfy_index.append(i)
        index = satisfy_index + unsatisfy_index

        for index_no in index:
            if self.checkKKT(index_no):
                continue
            else:
                E1 = self._E[index_no]
                if E1 >= 0:
                    j = min(range(self.data_size), key=lambda x: self._E[x])
                else:
                    j = max(range(self.data_size), key=lambda x: self._E[x])

            return index_no, j

    def SMO(self):
        '''
        SMO 算法
        :return:
        '''
        for iteration in range(self.iterations):
            print('第%d代' % iteration)
            index_1, index_2 = self.init_alpha()
            if self.trainY[index_1] == self.trainY[index_2]:
                L = max(0, self.alpha[index_1]+self.alpha[index_2]-self.penalty)
                H = min(self.penalty, self.alpha[index_1]+self.alpha[index_2])
            else:
                L = max(0, self.alpha[index_1] - self.alpha[index_2])
                H = min(self.penalty, self.penalty - self.alpha[index_1] + self.alpha[index_2])

            E1 = self._E[index_1]
            E2 = self._E[index_2]
            # eta=K11+K22-2K12
            eta = self.kernalFunction(
                self.trainX[index_1, :], self.trainX[index_1, :]
            ) + self.kernalFunction(
                self.trainX[index_2, :], self.trainX[index_2, :]
            ) - 2 * self.kernalFunction(
                self.trainX[index_1, :], self.trainX[index_2, :]
            )
            if eta <= 0:
                # print('eta <= 0')
                continue

            alpha2_new_unc = self.alpha[index_2] + self.trainY[index_2] * (E1 - E2) / eta

            if alpha2_new_unc > H:
                alpha2_new = H
            elif alpha2_new_unc < L:
                alpha2_new = L
            else:
                alpha2_new = alpha2_new_unc

            alpha1_new = self.alpha[index_1] + self.trainY[index_1] * self.trainY[index_2] * (
                    self.alpha[index_2] - alpha2_new
            )

            b1_new = -E1 - self.trainY[index_1] * self.kernalFunction(
                self.trainX[index_1], self.trainX[index_1]
            ) * (
                    alpha1_new - self.alpha[index_1]
            ) - self.trainY[index_2] * self.kernalFunction(
                self.trainX[index_2], self.trainX[index_1]
            ) * (alpha2_new - self.alpha[index_2]) + self.b

            b2_new = -E2 - self.trainY[index_1] * self.kernalFunction(
                self.trainX[index_1], self.trainX[index_2]
            ) * (
                    alpha1_new - self.alpha[index_1]
            ) - self.trainY[index_2] * self.kernalFunction(
                self.trainX[index_2], self.trainX[index_2]
            ) * (alpha2_new - self.alpha[index_2]) + self.b

            if 0 < alpha1_new < self.penalty:
                b_new = b1_new
            elif 0 < alpha2_new < self.penalty:
                b_new = b2_new
            else:
                # 选择中点
                b_new = (b1_new + b2_new) / 2

            # 更新参数
            self.alpha[index_1] = alpha1_new
            self.alpha[index_2] = alpha2_new
            self.b = b_new

            self._E[index_1] = self._getE(index_1)
            self._E[index_2] = self._getE(index_2)


def main():
    X, y = create_data()
    y = np.array([y]).T
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.25)
    clf = SVM(200)
    clf.fit(X_train, y_train)
    acc = clf.accurate(X_test, y_test)
    print(acc)

if __name__ == '__main__':
    main()