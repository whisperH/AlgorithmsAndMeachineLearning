import pandas as pd
from BoostModel.DecisionTree import *
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, [0, 1, -1]])
    for i in range(len(data)):
        if data[i, -1] == 0:
            data[i, -1] = -1
    # print(data)
    return data[:, :2], data[:, -1]

class AdaBoost(object):
    def __init__(self, n_estimator, weaker_learner, column_type):
        '''
        AdaBoost 模型初始化
        :param n_estimator: 选取n_estimator个弱学习器
        '''
        # 弱学习器的数量
        self.n_estimator = n_estimator
        # 每一列数据的类型（连续/离散）
        self.column_type = column_type
        # 初始化每个学习器权重
        self.alpha = [1] * n_estimator
        # 初始弱学习器数组
        self.weaker_learners = []

        # 初始化样本数据集大小以及特征数量
        self.data_size, self.feature_size = 0, 0

        # 特征的索引值
        self.feature_indices = []

        # 初始化弱学习器
        self.weaker_learner = weaker_learner
        # 初始化训练集的权值分布
        self.w = None

    def __initArgs__(self, _X, _Y):
        '''
        相关参数初始化
        :param _X: 训练集特征值
        :param _Y: 训练集标签值
        :return:
        '''
        self.train_x = _X
        self.train_y = _Y
        self.data_size, self.feature_size = _X.shape
        self.feature_indices = [_ for _ in range(self.feature_size)]

    def getErrorRate(self, true_y, fake_y):
        '''
        计算第ith_estimator个弱学习器的误差
        :param fake_y: 弱学习器的编号
        :return: 弱学习器的错误率
        '''
        aggErrors = np.multiply(np.mat(true_y) != np.mat(fake_y), np.ones(fake_y.shape))
        return aggErrors.sum() / true_y.shape[0]

    def getAlpha(self, error_rate):
        '''
        计算第ith_estimator个弱学习器的权重
        :param error_rate: 错误率
        :return: 弱学习器对应的权重
        '''

        return 0.5 * np.log((1-error_rate)/error_rate)

    def fit(self, _X, _Y):
        '''
        AdaBoost 模型训练
        :param _X: 特征值
        :param _Y: 标签值
        :return:
        '''
        self.__initArgs__(_X, _Y)
        for ith_estimator in range(self.n_estimator):
            print('%d\'s estimator:' % ith_estimator)

            # 初始化分布
            self.w = 1 / self.data_size * np.ones((self.data_size, 1))

            # 新建一个弱学习器
            # 寻找最优的划分节点
            weaker_learner = self.weaker_learner(ith_estimator, self.column_type)
            # split_feature_index, best_gini, best_theshold = weaker_learner.getBestFeature(_X, _Y, self.feature_indices)
            # train_x = _X[:, split_feature_index]
            # X = np.array([train_x]).T
            # print('best_theshold', best_theshold)
            # print('split_feature_index:', split_feature_index)
            # weaker_learner.column_type = [self.column_type[split_feature_index]]

            weaker_learner.fit(_X, _Y)
            print(weaker_learner.__root__.feature_index)

            weaker_learner_result = weaker_learner.predict(_X)

            # self.weaker_learners.append((split_feature_index, weaker_learner))
            self.weaker_learners.append((weaker_learner.__root__.feature_index, weaker_learner))

            # 计算错误率
            error_rate = self.getErrorRate(self.train_y, weaker_learner_result)
            print('error_rate:', error_rate)

            # 计算该弱学习器的权重
            ith_alpha = self.getAlpha(error_rate)
            print('alpha:', ith_alpha)
            self.alpha[ith_estimator] = ith_alpha

            print(self.train_y)
            print(weaker_learner_result)

            # 更新w
            w_tmp = - ith_alpha * np.multiply(self.train_y, weaker_learner_result)
            self.w = np.multiply(self.w, np.exp(w_tmp))
            print('update weights:', self.w)
            # 规范化w
            self.w /= self.w.sum()
            print('normal weights:', self.w)

            # 如果错误率比随机猜还差，那就停止
            if error_rate > 0.5:
                break
            else:
                _X, _Y = self.resampleData()
                print('resample data')
                print(_X)
                print(_Y)

        print('train done')

    def resampleData(self):
        '''
        数据重采样，先计算每个样本需要被抽取的次数，
        然后列索引按照抽取次数从大到小排序（注意是列）
        :return:
        '''
        # 确定每个样本需要抽取的次数

        nums = list(np.multiply(self.w.T[0], self.data_size))

        # 将权重的索引按 权重的大小排序（从大到小）
        idx = list(np.argsort(self.w.T[0]))
        idx.reverse()


        new_index = []
        for id in idx:
            num_arr = (int(nums[id]) + 1) * [id]
            new_index.extend(num_arr)
            if len(new_index) == self.data_size:
                break
        return self.train_x[new_index], self.train_y[new_index]

    def predict(self, X):
        '''
        预测
        :param x: 1*d 维的特征值
        :return: 预测结果
        '''
        print('predict')
        result = []
        for x in X:
            res = 0
            for index, weaker_learner in enumerate(self.weaker_learners):
                res += self.alpha[index] * weaker_learner[1].predict(
                    np.array([x])
                )
            result.append(1 if res > 0 else -1)
        return np.array([result]).T


def main():
    # X, y = create_data()
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # column_type = ['series', 'series']
    # clf = AdaBoost(3, DecisionStump, column_type)
    # clf.fit(X_train, np.array([y_train]).T)
    # print(clf.alpha)
    # y_fake = clf.predict(X_test)
    # print(np.array([y_test]).T)
    # print(y_fake)
    # print(y_fake - np.array([y_test]).T)
    # print(clf.getErrorRate(np.array([y_test]).T, y_fake))

    X = np.array([
        # ['yes'], ['no'], ['yes'], ['yes'], ['no'], ['no'], ['yes'], ['yes'],
        # ['yes'], ['yes'], ['no'], ['yes'], ['yes'], ['yes'], ['no'], ['yes'],
        # [205],   [180],   [210],   [167],   [156], [125],  [168],  [172]
        [1, 1, 205],
        [0, 1, 180],
        [1, 0, 210],
        [1, 1, 167],
        [0, 1, 156],
        [0, 1, 125],
        [1, 0, 168],
        [1, 1, 172]
    ])
    y = np.array([
        [1], [1], [1], [1], [-1], [-1], [-1], [-1]
    ])
    column_type = ['dispersed', 'dispersed', 'series']
    clf = AdaBoost(3, DecisionStump, column_type)
    clf.fit(X, y)
    y_fake = clf.predict(X)
    print(clf.alpha)
    print(clf.getErrorRate(y, y_fake))

    # X = np.arange(10).reshape(10, 1)
    # y = np.array([[1], [1], [1], [-1], [-1], [-1], [1], [1], [1], [-1]])
    # column_type = ['series']



if __name__ == '__main__':
    main()