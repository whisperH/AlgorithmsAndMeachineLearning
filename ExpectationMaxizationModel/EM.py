import numpy as np
from scipy import stats

def generateData():
    '''
    根据多元高斯分布需要的配置，均值、协方差矩阵、生成数量等生成数据
    :return: 特征值和标签值
    '''
    np.random.seed(0)
    # 各个维度的均值
    mu1_fact = (0, 0, 0)
    # 协方差矩阵
    cov_fact = np.identity(3)
    positive_data = np.random.multivariate_normal(mu1_fact, cov_fact, 400)

    # 各个维度的均值
    mu2_fact = (2, 2, 1)
    # 协方差矩阵
    cov_fact = np.identity(3)
    negative_data = np.random.multivariate_normal(mu2_fact, cov_fact, 100)
    X = np.vstack((positive_data, negative_data))
    y = np.array([True] * 400 + [False] * 100)

    return X, y


class EM_Model(object):
    def __init__(self, distribution_config: dict, iterations: int):
        '''
        初始化模型，高斯分布需要的参数，均值，协方差矩阵，各个分布的概率
        :param distribution_config:
        :param iterations:
        '''
        self.distribution_nums = len(distribution_config)
        self.distribution_info = {}
        self.iterations = iterations

        for key, value in distribution_config.items():
            print('initiate distribution:', key)
            self.distribution_info[key] = {
                # 该分布的先验概率
                'priority_prob': value['priority_prob'],
                # 该分布的均值
                'mean': value['mean'],
                # 该分布的协方差矩阵
                'cov': value['cov'],
                # 该分布的样本概率
                'sample_prob': 0,
                # 该分布的后验概率
                'posterior_prob': 0,
            }

        self.data_size = 0
        self.features = 0

    def __initArgs__(self, X, y):
        '''
        初始化数据
        :param X: 特征值
        :param y: 标签纸
        :return:
        '''
        self.data_size, self.features = X.shape

    def fit(self, X, y):
        '''
        拟合数据
        :param X: 特征值
        :param y: 标签值
        :return:
        '''
        self.__initArgs__(X, y)
        for iteration in range(self.iterations):
            # 1.计算各个分布所形成的后验概率
            # 1.1 K 个分布样本的概率和
            all_K_prob = 0
            for k_distribution in self.distribution_info:
                mean_value = self.distribution_info[k_distribution]['mean']
                cov_value = self.distribution_info[k_distribution]['cov']
                priority_prob = self.distribution_info[k_distribution]['priority_prob']
                # 每个样本中第k_distribution个分布所占的比重(m*1)
                y_k = stats.multivariate_normal.pdf(
                    X,
                    mean=mean_value,
                    cov=cov_value,
                )
                # 将维度为(m,)转成(m,1)
                y_k = np.array([y_k]).T
                # K个分布样本概率和(所有样本 m*1)
                all_K_prob += y_k * priority_prob

                # 更新参数
                self.distribution_info[k_distribution]['sample_prob'] = y_k

            # 1.2 计算各个分布所形成的后验概率
            for k_distribution in self.distribution_info:
                sample_prob = self.distribution_info[k_distribution]['sample_prob']
                priority_prob = self.distribution_info[k_distribution]['priority_prob']

                # 第k_distribution个分布的后验概率（所有样本 m*1）
                posterior_prob = sample_prob * priority_prob / all_K_prob
                self.distribution_info[k_distribution]['posterior_prob'] = posterior_prob

            # 2. 更新参数
            for k_distribution in self.distribution_info:
                posterior_prob = self.distribution_info[k_distribution]['posterior_prob']

                gamma_k = posterior_prob.sum(axis=0)
                # 更新第k_distribution分布的均值
                new_mean_value = (posterior_prob * X).sum(axis=0) / gamma_k

                # 更新第k_distribution分布的协方差矩阵
                new_cov_value = np.zeros(self.distribution_info[k_distribution]['cov'].shape)
                for i in range(self.data_size):
                    i_posterior_prob = posterior_prob[i]
                    new_cov_value += i_posterior_prob * np.dot(
                        (X[i] - np.tile(new_mean_value, (1, 1))).T,
                        (X[i] - np.tile(new_mean_value, (1, 1)))
                    )
                new_cov_value /= gamma_k

                # 更新参数
                priority_prob = gamma_k / self.data_size

                self.distribution_info[k_distribution]['mean'] = new_mean_value
                self.distribution_info[k_distribution]['cov'] = new_cov_value
                self.distribution_info[k_distribution]['priority_prob'] = priority_prob
        print('train done!')

    def predict(self, data):
        '''
        预测分类
        :param data:
        :return:
        '''
        predict = []
        for k_distribution in self.distribution_info:
            norm1 = stats.multivariate_normal(
                self.distribution_info[k_distribution]['mean'],
                self.distribution_info[k_distribution]['cov']
            )
            predict.append(norm1.pdf(data))

def main():
    X, y = generateData()
    print(X)
    exit(32)
    distribution_config = {
        '1': {
            'mean': X.min(axis=0),
            'cov': np.identity(X.shape[1]),
            'priority_prob': 0.5
        },
        '2': {
            'mean': X.max(axis=0),
            'cov': np.identity(X.shape[1]),
            'priority_prob': 0.5
        }
    }
    print(distribution_config)
    clf = EM_Model(distribution_config, 100)
    clf.fit(X, y)
    print(clf.distribution_info['1']['mean'])
    print(clf.distribution_info['1']['cov'])
    print(clf.distribution_info['1']['priority_prob'])
    print(clf.distribution_info['2']['mean'])
    print(clf.distribution_info['2']['cov'])
    print(clf.distribution_info['2']['priority_prob'])


if __name__ == '__main__':
    main()