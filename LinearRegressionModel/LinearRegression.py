import pandas as pd
import numpy as np

class Regression(object):
    def __init__(self):
        self.ws = None

    def standRgressFit(self, x_data, y_data):
        '''
        标准线性回归
        :param data: 待拟合的数据
        :param x_name: 输入的特征名称
        :param y_name: 拟合对象的特征名称
        :return: 系数
        '''
        x_matrix = np.mat(x_data.values)
        y_matrix = np.mat(y_data.values)

        xTx = x_matrix.T * x_matrix
        # 如果是非满秩矩阵
        if np.linalg.det(xTx) == 0:
            print("This matrix is singular, cannot do inverse")
            return
        else:
            self.ws = xTx.I * (x_matrix.T * y_matrix)

    def lwlrFit(self, test_point, x_data, y_data, k=1):
        '''
        局部加权线性回归
        :param test_point: 待预测点的特征
        :param x_data: 训练集的特征数据
        :param y_data: 训练集的结果数据
        :param k: 权重的宽度
        :return: 加权线性回归的权重矩阵
        '''
        xMat = np.mat(x_data)
        yMat = np.mat(y_data).T
        m = np.shape(xMat)[0]
        # 创建一个m*m的单位矩阵
        weights = np.mat(np.eye((m)))
        # 对角矩阵赋予高斯核函数
        for j in range(m):  # next 2 lines create weights matrix
            diffMat = test_point - xMat[j, :]  #
            weights[j, j] = np.exp(diffMat * diffMat.T / (-2.0 * k ** 2))
        xTx = xMat.T * (weights * xMat)
        if np.linalg.det(xTx) == 0.0:
            print("This matrix is singular, cannot do inverse")
            return
        self.ws = xTx.I * (xMat.T * (weights * yMat))

    def ridgeFit(self, x_data, y_data, lam=0.2):
        '''
        岭回归拟合
        :param x_data: 训练数据的特征值
        :param y_data: 训练数据的y值
        :param lam: lambda的取值
        :return: 线性回归的参数
        '''
        x_matrix = np.matrix(x_data)
        y_matrix = np.matrix(y_data)

        denom = x_matrix.T * x_matrix + lam * np.eye(x_matrix.shape[1])
        if np.linalg.det(denom) == 0:
            print("This matrix is singular, cannot do inverse")
            return
        else:
            self.ws = denom.I * (x_matrix.T * y_matrix)

    def stageWiseFit(self, x_data, y_data, epsilon=0.01, iterations=200):
        x_matrix = np.matrix(x_data)
        y_matrix = np.matrix(y_data)

        weight_num = x_data.shape[1]
        # 初始化权重
        self.ws = np.zeros((weight_num, 1))
        bset_w = self.ws.copy()
        # 初始化权重的记录矩阵
        # （用returnMat矩阵记录iterations次迭代中的权重
        returnMat = np.zeros((iterations, weight_num))  # testing code remove

        for i in range(iterations):
            lowest_error = np.inf
            # 修改第j个特征的权重
            for j in range(weight_num):
                # sign 决定方向，往哪边走
                for sign in [-1, 1]:
                    current_w = self.ws.copy()
                    current_w[j] += epsilon * sign
                    y_hat = x_matrix * current_w

                    # 修改第j个特征后，误差是否有所降低？
                    current_error = rssError(y_matrix.A, y_hat.A)

                    if lowest_error > current_error:
                        lowest_error = current_error
                        bset_w = current_w
            self.ws = bset_w
            returnMat[i, :] = self.ws.T
        return returnMat

    def predict(self, x_data):
        '''
        预测数据
        :param x_data: 预测的特征值
        :return: y模拟值
        '''
        x_matrix = np.mat(x_data.values)
        return x_matrix * self.ws

def rssError(yArr,yHatArr): #yArr and yHatArr both need to be arrays
    return ((yArr-yHatArr)**2).sum()

def corrCoef(y_hat, y):
    '''
    计算相关系数
    :param y_hat: 模拟值
    :param y: 真实值
    :return: 相关系数矩阵
    '''
    return np.corrcoef(y_hat.T, y.T)

def regularize(data):
    '''
    标准化数据
    :param data: 待标准化的dataframe
    :return: 标准化后的dataframe
    '''
    data = (data - data.mean()) / data.var()
    return data

if __name__ == '__main__':
    data = pd.read_table('data/abalone.txt', sep='\t')
    x_features = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7']
    y_features = ['y']

    data = regularize(data)

    rg = Regression()
    weight_matrix = rg.stageWiseFit(
        data[x_features].values, data[y_features].values,
        0.01, 200
    )
    print(rg.ws)
    print(weight_matrix)

    # print(corrcoef(rg.predict(data[x_features]), data[y_features]))