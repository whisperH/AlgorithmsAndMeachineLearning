import pandas as pd
from DecisionTreeModel.visualization import *
import math

def getWaterMelonData():
    '''
    获取西瓜数据集2.0
    :return: 数据集，特征名称，标签名称
    '''
    dataset = pd.read_csv('./data/WaterMelonDataset2.csv')
    features_name = [
        '色泽', '根蒂', '敲声', '纹理', '脐部', '触感'
    ]
    label_names = '好瓜'
    return dataset, features_name, label_names




class DecisionTree(object):
    def __init__(self, dataset, features_name, label_name):
        '''
        初始化决策树
        :param dataset: 待处理的数据集
        :param features_name: 特征名称
        :param label_names: 标签名称
        '''
        self.dataset = dataset
        self.features_name = features_name
        self.label_name = label_name
        # 计算特征与标签的所有特征值
        self.unique_value = {}
        self.getUniqueValue()

    def getUniqueValue(self):
        '''
        计算原始数据集中所有特征值
        :return:
        '''
        for feaure_name in self.features_name:
            self.unique_value[feaure_name] = list(dataset[feaure_name].unique())


    def createDecisionTree(self, dataset, features_name, method='gain'):
        '''
        构造决策树
        :param dataset: 构造树需要用到的数据集
        :param features_name: 构造树需要用到的特征名称
        :param method: 判定最优特征的方法
        :return: 当前生成的子树节点
        '''
        # 选出当前数据集中所有类的取值
        label_values = list(dataset[self.label_name].unique())

        # 选出当前数据集中最多的类
        max_label = None
        max_count = 0
        for label_value in label_values:
            label_num = dataset[dataset[self.label_name] == label_value].shape[0]
            if label_num > max_count:
                max_label = label_value
                max_count = label_num

        # 如果数据集中的类标签只有一种，则返回该标签
        if len(label_values) == 1:
            return max_label
        # 如果待分类的特征为空 或者 检查数据集中的特征种类是否为1（数据集中所有特征都相同）
        elif len(features_name) == 0 or dataset.drop_duplicates(subset=features_name, keep=False).empty:
            # 选取当前数据集中最多的标签作为该类标签
            return max_label


        # 选出最佳的分类特征
        classify_feature = self.classifyMethod(dataset, feature_names, method=method)
        # 初始化节点
        DTree = {classify_feature: {}}

        # feature_values = list(dataset[classify_feature].unique())
        feature_values = self.unique_value[classify_feature]
        # 对数据集进行分类
        for feature_value in feature_values:
            sub_dataset = dataset[dataset[classify_feature] == feature_value]

            # 如果数据集为空
            if sub_dataset.shape[0] == 0:
                DTree[classify_feature][feature_value] = max_label
            else:
                # 将特征复制一份，传递给子数据集
                sub_feature_names = features_name.copy()
                sub_feature_names.remove(classify_feature)

                DTree[classify_feature][feature_value] = self.createDecisionTree(
                    sub_dataset, sub_feature_names, method=method
                )
        return DTree

    def classifyMethod(self, dataset, feature_names, method='gain'):
        '''
        划分特征值的标准，默认选择为信息增益划分
        :param method: 划分特征值的标准
        :return:
        '''
        classify_feature = {}
        # 如果是信息增益方式的话，选择信息增益最大的特征
        if method == 'gain':
            columns = ['gain']
            for feature_name in feature_names:
                classify_feature[feature_name] = calculateGain(
                    dataset, feature_name, self.label_name
                )
            classify_feature = pd.DataFrame(classify_feature, index=columns).T
            print(classify_feature)
            # 返回最大值的索引（特征名称）
            return classify_feature.idxmax().values[0]

        # 先选出信息增益高于平均水平的属性，再从中选择增益率最高的特征
        elif method == 'gain_ratio':
            columns = ['gain_ratio', 'IV', 'gain']
            for feature_name in feature_names:
                classify_feature[feature_name] = calculateGainRatio(
                    dataset, feature_name, self.label_name
                )
            classify_feature = pd.DataFrame(classify_feature, index=columns).T
            # 选出信息增益高于平均水平的属性
            high_gain_dataset = classify_feature[classify_feature['gain'] > classify_feature['gain'].mean()]
            # 从中选择增益率最高的特征
            return high_gain_dataset[['gain_ratio']].idxmax().values[0]

        # 选出基尼指数最小的属性
        elif method == 'gini':
            columns = ['gini']
            for feature_name in feature_names:
                classify_feature[feature_name] = calculateGiniIndex(
                    dataset, feature_name, self.label_name
                )
            classify_feature = pd.DataFrame(classify_feature, index=columns).T
            # print(classify_feature)
            return classify_feature[['gini']].idxmin().values[0]



def calculateEnt(dataset, label_name):
    '''
    计算dataset的信息熵: Ent(D)=-\sum_{k=1}^{|Y|} \frac{1}{|y|}log_2\frac{1}{|y|}=log_2|y|
    :param dataset:特征数据信息
    :param label_name: 标签名称
    :return:信息熵
    '''
    # 列举出类标签所有可能的取值
    label_values = list(dataset[label_name].unique())
    Ent = 0
    for label_value in label_values:
        tmp_data = dataset[dataset[label_name] == label_value]
        # 计算某一类的概率
        prob = tmp_data.shape[0] / dataset.shape[0]
        Ent += prob * math.log2(prob)
    return -Ent

def calculateGain(dataset, feature_name, label_name):
    '''
    计算dataset中特征为feature_name的信息增益
    :param dataset:数据信息
    :param label: 标签数据
    :return:信息增益，以及对应的条件熵
    '''
    # 计算类标签的信息熵
    Ent = calculateEnt(dataset, label_name)
    # 计算特征为feature_name的条件信息熵
    feature_values = list(dataset[feature_name].unique())
    condition_Ent = 0
    for feature_value in feature_values:
        # 选出特征feature_name中特征值为feature_value的数据
        sub_data = dataset[dataset[feature_name] == feature_value]
        # 计算概率
        prob = sub_data.shape[0] / dataset.shape[0]
        condition_Ent += prob * calculateEnt(sub_data, label_name)
    return Ent - condition_Ent

def calculateGainRatio(dataset, feature_name, label_name):
    '''
    计算dataset中特征为feature_name的信息增益
    :param dataset:数据信息
    :param feature_values:数据信息
    :param label_name: 标签数据
    :return:信息增益
    '''
    # 计算信息增益
    Gain = calculateGain(dataset, feature_name, label_name)
    # 计算特征为feature_name的IV值
    feature_values = list(dataset[feature_name].unique())
    IV = 0
    for feature_value in feature_values:
        # 选出特征feature_name中特征值为feature_value的数据
        sub_data = dataset[dataset[feature_name] == feature_value]
        # 计算概率
        prob = sub_data.shape[0] / dataset.shape[0]
        IV += prob * math.log2(prob)

    return -Gain/IV, -IV, Gain

def calculateGini(dataset, label_name):
    '''
    计算dataset中特征为feature_name的基尼系数
    :param dataset:数据信息
    :param label_name: 标签数据
    :return:基尼系数
    '''
    label_values = list(dataset[label_name].unique())
    condition_prob = 0
    for label_value in label_values:
        # 选出特征feature_name中特征值为feature_value的数据
        sub_data = dataset[dataset[label_name] == label_value]
        # 计算概率
        condition_prob += (sub_data.shape[0] / dataset.shape[0])**2
    return 1 - condition_prob

def calculateGiniIndex(dataset, feature_name, label_name):
    '''
    计算dataset中特征为feature_name的基尼系数
    :param dataset:数据信息
    :param feature_values:数据信息
    :param label_name: 标签数据
    :return:基尼系数
    '''
    feature_values = list(dataset[feature_name].unique())
    Gini_index = 0
    for feature_value in feature_values:
        # 选出特征feature_name中特征值为feature_value的数据
        sub_data = dataset[dataset[feature_name] == feature_value]
        Gini = calculateGini(sub_data, label_name)
        # 计算概率
        Gini_index += (sub_data.shape[0] / dataset.shape[0]) * Gini
    return Gini_index

if __name__ == '__main__':
    dataset, feature_names, label_names = getWaterMelonData()
    DTObject = DecisionTree(dataset, feature_names, label_names)
    # DTObject.classifyMethod(dataset, feature_names, 'gain')
    DTObject.root = DTObject.createDecisionTree(dataset, feature_names, 'gain')
    print(DTObject.root)
    createPlot(DTObject.root)
