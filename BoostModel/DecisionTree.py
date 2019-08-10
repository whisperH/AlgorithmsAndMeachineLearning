import numpy as np

class Node(object):
    def __init__(self, X=None, Y=None, max_label=None):
        # 存储当前节点训练的数据集
        self.X = X
        self.Y = Y
        # 存储当前节点对应数据集中数量最多的标签，后剪枝时可调用
        self.max_label = max_label
        # 存储当前节点划分的特征下标
        self.feature_index = None
        # 存储当前节点的划分结果
        self.next = {}
        # 存储当前节点信息损失等的计算结果
        self.LossInfo = None

class DecisionTree(object):
    def __init__(self, column_type):
        self.column_type = column_type
        # self.column_type = ['series']
        self.data_size, self.feature_nums = 0, 0
        self.__root__ = None
        self.feature_indices = []

    def __initArgs__(self, _X, _Y):
        self.train_x = _X
        self.train_y = _Y
        self.data_size, self.feature_nums = _X.shape
        self.feature_values = {}

        # 特征的索引值
        self.feature_indices = [_ for _ in range(self.feature_nums)]

        # 记录每一个特征可能的取值
        for i in self.feature_indices:
            self.feature_values[i] = None
            # 若是连续值，则记录最佳分隔点，若是离散值则存储其特征取值
            if not self.isSeries(self.train_x[:, i], i):
                self.feature_values[i] = np.unique(self.train_x[:, i])
            else:
                best_gini_index, best_threshold = getSeriesGiniIndex(
                    self.train_x, i, self.train_y
                )
                self.feature_values[i] = best_threshold

    def getMaxLabel(self, _Y):
        '''
        计算 _Y 数据集中数量最多的标签值
        :param _Y: 标签值
        :return:
        '''
        labels = np.unique(_Y)
        max_count = 0
        max_label = None
        for ilabel in labels:
            ilabel_nums = len(_Y[np.where(_Y==ilabel)])
            if max_count < ilabel_nums:
                max_count = ilabel_nums
                max_label = ilabel
        return max_label

    def createDecisionTree(self, X, Y, feature_indices):
        '''
        创建决策树
        :param X: 特征值
        :param Y: 标签值
        :param feature_indices: 特征索引值
        :return:
        '''
        # bincount统计每个元素出现的次数，argmax选出最大的
        max_label = self.getMaxLabel(Y)
        # 创建一个新节点
        new_node = Node(X, Y, max_label)
        if len(np.unique(Y)) == 1:
            return max_label
        if len(feature_indices) == 0 or len(np.unique(X, axis=0)) == 1:
            return max_label
        else:
            split_feature_index, best_gini, best_theshold = self.getBestFeature(X, Y, feature_indices)

            new_node.feature_index = split_feature_index
            new_node.LossInfo = best_gini
            # 存储当前节点划分的特征名称
            new_node.next = {}
            # 如果 best_theshold 为 None，说明为离散值
            if not self.isSeries(X[:, new_node.feature_index], new_node.feature_index):
                for feature_value in self.feature_values[new_node.feature_index]:
                    new_node = self.__createDecisionNode__(X, Y, feature_value, new_node, feature_indices, inequal_flag='=')
            # 说明为连续值
            else:
                print('Series data threshold:', best_theshold)
                # 先取小于阈值的数据
                new_node = self.__createDecisionNode__(X, Y, best_theshold, new_node, feature_indices, inequal_flag='<=')
                # 大于阈值的数据
                new_node = self.__createDecisionNode__(X, Y, best_theshold, new_node, feature_indices, inequal_flag='>')
        return new_node

    def __createDecisionNode__(self, X, Y, feature_value, node, feature_indices, inequal_flag='='):
        '''
        创建决策树节点
        :param X: 特征值
        :param Y: 标签值
        :param feature_value: 特征中的某一取值，用节点的名称
        :param node: 初始化后的节点
        :param feature_indices: 特征索引值
        :param inequal_flag: 子数据集选择条件
        :return: 完整的决策节点
        '''
        # 设置划分结点的名称
        if inequal_flag == '=':
            idx = np.where(X[:, node.feature_index] == feature_value)
            node_name = feature_value
        elif inequal_flag == '>':
            idx = np.where(X[:, node.feature_index] > feature_value)
            node_name = '>' + str(feature_value)
        elif inequal_flag == '<=':
            idx = np.where(X[:, node.feature_index] <= feature_value)
            node_name = '<=' + str(feature_value)

        # 划分数据集
        sub_label = Y[idx]
        sub_data = X[idx]
        if len(feature_indices) == 0:
            node.next[node_name] = node.max_label
        else:
            sub_feature_indices = feature_indices.copy()
            sub_feature_indices.remove(node.feature_index)
            node.next[node_name] = self.createDecisionTree(sub_data, sub_label, sub_feature_indices)
        return node

    def getBestFeature(self, X, Y, feature_indices):
        '''
        获取最佳分隔特征
        :param X: 特征值
        :param Y: 标签值
        :param feature_indices: 特征的索引数组
        :return: 分隔特征的编号， 基尼值， 如果是连续性数据，还有分隔阈值
        '''
        best_gini = np.inf
        split_feature_index = np.inf
        best_theshold = None

        for i in feature_indices:
            # print('feature:', i)
            if self.isSeries(X[:, i], i):
                gini_index, best_theshold = getSeriesGiniIndex(X, i, Y)
            else:
                gini_index, best_theshold = getDispersedGiniIndex(X, i, Y)
            # print('gini_index', gini_index)
            if gini_index < best_gini:
                best_gini = gini_index
                split_feature_index = i
        return split_feature_index, best_gini, best_theshold

    def fit(self, X, Y):
        self.__initArgs__(X, Y)
        self.__root__ = self.createDecisionTree(X, Y, self.feature_indices)


    def isSeries(self, column_data, index):
        '''
        判断 column_data 是否是连续数据
        :param column_data: 列数据
        :param index: 列索引
        :return:
        '''
        # print(column_data)
        # if column_data.dtype in series_type:
        #     return True
        # else:
        #     return False
        if self.column_type[index] == 'series':
            return True
        else:
            return False

    def predict(self, dataset):
        '''
        预测dataset的标签值
        :param dataset:
        :return:
        '''
        result = []
        for data in dataset:
            node = self.__root__

            while type(node).__name__ == 'Node':
                # 从决策树中取出特征
                feature_index = node.feature_index
                # 如果测试数据中有该特征,
                # 先从测试数据中取出对应的特征值，再在决策树中找相应的子树
                if feature_index in self.feature_values:
                    if self.isSeries(data[feature_index], feature_index):
                        if data[feature_index] <= self.feature_values[feature_index]:
                            node = node.next['<='+str(self.feature_values[feature_index])]
                        else:
                            node = node.next['>'+str(self.feature_values[feature_index])]
                    else:
                        node = node.next[data[feature_index]]
            result.append(node)
        return np.array([result]).T

class DecisionStump(DecisionTree):
    def __init__(self, learner_name, column_type):
        '''
        该类为决策树桩，继承于决策树
        :param learner_name:
        :param column_type:
        '''
        super().__init__(column_type)
        self.learner_name = learner_name


    def fit(self, X, Y):
        # 决策树初始化
        self.__initArgs__(X, Y)
        # 寻找最优的划分节点
        split_feature_index, best_gini, best_theshold = self.getBestFeature(X, Y, self.feature_indices)
        feature_indices = [split_feature_index]
        print('best_theshold', best_theshold)
        print('split_feature_index:', split_feature_index)
        self.__root__ = self.createDecisionStump(X, Y, feature_indices, best_theshold)
        self.__root__.LossInfo = best_gini

    def createDecisionStump(self, X, Y, feature_indices, best_theshold):
        # bincount统计每个元素出现的次数，argmax选出最大的
        max_label = self.getMaxLabel(Y)
        # 创建一个新节点
        new_node = Node(X, Y, max_label)

        new_node.feature_index = feature_indices[0]

        # 存储当前节点划分的特征名称
        new_node.next = {}
        # 如果 best_theshold 为 None，说明为离散值
        if not self.isSeries(X[:, new_node.feature_index], new_node.feature_index):
            for feature_value in self.feature_values[new_node.feature_index]:
                new_node = self.__createDecisionNode__(X, Y, feature_value, new_node, feature_indices,
                                                       inequal_flag='=')
        # 说明为连续值
        else:
            print('Series data threshold:', best_theshold)
            # 先取小于阈值的数据
            new_node = self.__createDecisionNode__(X, Y, best_theshold, new_node, feature_indices,
                                                   inequal_flag='<=')
            # 大于阈值的数据
            new_node = self.__createDecisionNode__(X, Y, best_theshold, new_node, feature_indices, inequal_flag='>')
        return new_node

def getGiniValue(label_data):
    '''
    计算dataset的基尼值: Gini(D)=1-\sum_{k=1}^{|Y|} p_k^2
    :param dataset:特征数据
    :param label: 标签数据
    :return:基尼值
    '''
    # 列举出类标签所有可能的取值
    label_values = np.unique(label_data)
    gini = 0
    for label_value in label_values:
        tmp_data = label_data[np.where(label_data == label_value)]
        # 计算某一类的概率
        prob = len(tmp_data) / label_data.shape[0]
        gini += prob ** 2
    return 1 - gini

def getSeriesGiniIndex(data, column_index, label):
    column_data = data[:, column_index].astype(np.int)
    # 这里数据类型需要强制设置为float型
    # column_data = np.array([column_data]).T

    best_gini_index = np.inf
    best_threshold = np.inf
    # 先对数据排序，按照顺序重组特征值和标签值
    idx = np.argsort(column_data)
    column_data = column_data[idx]
    label = label[idx]

    # 计算相邻点的中位数，二分数据
    tmp = np.roll(column_data, -1)
    thresholds = (column_data + tmp) / 2
    # 计算不同组的信息损失

    for feature_threshold in thresholds[: -1]:
        # 筛选数据
        sub_data = column_data[np.where(column_data < feature_threshold)]
        sub_label = label[np.where(column_data < feature_threshold)]
        # 计算概率
        prob = sub_data.shape[0] / column_data.shape[0]
        gini_index = prob * getGiniValue(sub_label)

        # 筛选数据
        sub_data = column_data[np.where(column_data > feature_threshold)]
        sub_label = label[np.where(column_data > feature_threshold)]
        # 计算概率
        prob = sub_data.shape[0] / column_data.shape[0]
        gini_index += prob * getGiniValue(sub_label)

        # 寻找最佳分隔中位数
        if gini_index < best_gini_index:
            best_gini_index = gini_index
            best_threshold = feature_threshold
    # print('best_threshold is :', best_threshold)

    return best_gini_index, best_threshold

def getDispersedGiniIndex(data, column_index, label):
    column_data = data[:, column_index]
    gini_index = 0
    # 如果数据是离散的
    feature_values = np.unique(column_data)
    for feature_value in feature_values:
        # 选出特征 dataset 中特征值为feature_value的数据
        sub_data = column_data[np.where(column_data == feature_value)]
        sub_label = label[np.where(column_data == feature_value)]
        # 计算概率
        prob = len(sub_data) / column_data.shape[0]
        # 计算基尼指数
        gini_index += prob * getGiniValue(sub_label)
    return gini_index, None



def test():
    # 还是载入数据时数据类型的问题
    data = np.array([
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
    label = np.array([
        [1], [1], [1], [1], [0], [0], [0], [0]
    ])
    # gini_index = getSeriesGiniIndex(data, 0, label)
    # print(gini_index)
    # data = np.arange(10).reshape(10, 1)
    # label = np.array([[1], [1], [1], [-1], [-1], [-1], [1], [1], [1], [-1]])
    node = DecisionStump('name1', ['dispersed', 'dispersed', 'series'])
    node.fit(data, label)
    # print(node.__root__.next)
    print(node.predict(np.array([[1, 1, 167]])))


if __name__ == '__main__':
    test()