import numpy as np

class Perception(object):
    def __init__(self, input_vecs, labels, activation, lr=0.1):
        '''
        感知机模型权重以及偏置项的初始化
        :param input_shape: <tuple>需要输入层维度信息
        :param activation: <funciton>激活函数
        :param lr: <float>学习率
        '''
        self.input_vecs = input_vecs
        self.labels = labels
        self.n_features = input_vecs.shape[1]
        self.n_nums = input_vecs.shape[0]
        self.activation = activation
        self.weight = np.zeros((1, self.n_features))
        self.bias = np.zeros((1, 1))
        self.lr = lr

    def predict(self, input_vec):
        '''
        输入感知机运算结果
        :param input_vecs: <np.ndarray>训练数据
        :return:对ndarray中的逐元素应用激活函数
        '''
        output = np.dot(input_vec, self.weight.T) + self.bias
        return np.apply_along_axis(self.activation, axis=1, arr=output)

    def _update_weight(self, index_nums, delta):
        '''
        更新权重
        delta_weights = lr * (t - y) * x_i
        w是与x_i输入对应的权重项, t是训练样本的实际值, y是感知器的输出值，lr是学习速率
        ------
        :param input_vecs: <np.ndarray>训练数据
        :param delta: <np.ndarray>误差项
        :return:
        '''
        delta_weight = input_vecs[index_nums].T * delta
        self.weight += delta_weight

    def _update_bias(self, delta):
        '''
        更新偏差
        delta_bias = lr * (t - y)
        :param delta: <np.ndarray>误差项
        :return:
        '''
        self.bias += delta

    def forward(self, nums):
        '''
        前向计算感知机的输出值
        :param nums: 训练样例的数量
        :param input_vecs: 训练样本的特征值
        :return:
        '''
        for k in range(nums):
            print('%d th iterations' % k)
            for inums in range(self.n_nums):
                output = self.predict(self.input_vecs[inums])
                delta = self.lr * (self.labels[inums] - output)
                self._update_weight(inums, delta)
                self._update_bias(delta)
            # print('weights;', self.weight)
            # print('bias;', self.bias)
            print('output:', self.predict(self.input_vecs))


class PerceptionDualModel(object):
    def __init__(self, input_vecs, labels, activation, lr=1):
        '''
        感知机模型权重以及偏置项的初始化
        :param input_vecs: <np.ndarray>输入层的特征值
        :param labels: <np.ndarray>输入层对应的标签
        :param activation: <funciton>激活函数
        :param lr: <float>学习率
        '''
        self.input_vecs = input_vecs
        self.labels = labels

        self.activation = activation
        # 训练集特征的数量
        self.n_features = input_vecs.shape[1]
        # 训练集的数量
        self.n_nums = input_vecs.shape[0]
        self.lr = lr

        # n_i代表第i类数据使用的次数
        self.n = np.zeros((self.n_nums, 1))
        self.bias = 0

        # 计算Gram矩阵，为权重训练做准备
        self.Gram_metrix = self.calculate_Gram()

    def calculate_weights(self):
        '''
        根据公式计算权重
        weights = \sum_{i=1}^{N} n_i * lr * xi * yi
        :return:
        '''
        tmp = np.multiply(self.lr * self.n, self.labels)
        self.weight = (np.dot(tmp.T, self.input_vecs)).T

    def calculate_bias(self):
        '''
        计算偏差
        bias = \sum_{i=1}^{N} n_i * lr * yi
        :return:
        '''
        self.bias = np.dot(self.lr * self.n.T, self.labels)

    def calculate_Gram(self):
        return np.dot(self.input_vecs, self.input_vecs.T)

    def train(self, num_index):
        '''
        对感知机的权重进行驯良
        :param num_index: <int>输入数据的行标签，同时也意味着第几类数据
        :return:对ndarray中的逐元素应用激活函数
        '''

        output = np.dot(
            self.Gram_metrix[num_index],
            np.multiply(self.lr * self.n, self.labels)
        ) + self.bias
        return self.activation(output)

    def _update_weight(self, delta, num_index):
        '''
        更新权重
        n_i * lr = n_i * lr * + lr
        w是与x_i输入对应的权重项, t是训练样本的实际值, y是感知器的输出值，lr是学习速率
        ------
        :param delta: <np.ndarray>误差项
        :param num_index: <int>输入数据的行标签，同时也意味着第几类数据
        :return:
        '''
        # if delta == 0:
        #     print(num_index, ':分类正确')
        # else:
        #     print(num_index, ':分类错误')
        self.n[num_index] += delta

    def _update_bias(self, delta):
        '''
        更新偏差
        delta_bias = lr * y_i
        :param delta: <np.ndarray>误差项
        :return:
        '''
        self.bias += delta * self.lr

    def _predict(self, input_vec):
        '''
        根据特征值计算分类标签
        :param input_vec: 输入特征值
        :return:
        '''
        self.calculate_weights()
        # print('weights;', self.weight)
        # print('bias;', self.bias)
        predict = np.dot(input_vec, self.weight) + self.bias
        return np.apply_along_axis(self.activation, axis=1, arr=predict)


    def forward(self, nums):
        '''
        前向计算感知机的输出值
        :param nums: 迭代次数
        :param input_vecs: 训练样本的特征值
        :param labels: 训练样本的真实值
        :return:
        '''
        for k in range(nums):
            print('%d th iterations' % k)
            for num_index in range(self.n_nums):
                output = self.train(num_index)
                delta = self.labels[num_index] - output
                self._update_weight(delta, num_index)
                self._update_bias(delta)
            print('n;', self.n.T)
            print('bias;', self.bias)
            print(self._predict(self.input_vecs))
            print('===================')



def activation(x):
    '''
    对x中的所有元素逐元素的进行操作
    :param x:
    :return:
    '''
    return 1 if x > 0 else 0

def getTrainData():
    '''
    得到输入数据和输出数据
    :return:
        input_vecs<ndarray>：输入数据
        labels<ndarray>：标签
    '''
    # 构建训练数据
    # 输入向量列表
    # input_vecs = np.array([[1, 1], [0, 0], [1, 0], [0, 1]])
    # and
    # labels = np.array([1, 0, 0, 0])
    # or
    # labels = np.array([1, 0, 1, 1])

    input_vecs = np.array([[3, 3], [4, 3], [1, 1]])
    labels = np.array([[1], [1], [0]])
    return input_vecs, labels

if __name__ == '__main__':
    input_vecs, labels = getTrainData()
    # ================ 基本模式 ================#
    P = Perception(input_vecs, labels, activation, lr=1)
    P.forward(10)
    print(P.predict(np.array([[8, 6]])))

    # ================ 对偶模式 ================#
    P = PerceptionDualModel(input_vecs, labels, activation)
    P.forward(30)
    print(P._predict(np.array([[8, 6]])))
