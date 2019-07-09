# 文章说明：https://www.jianshu.com/p/2ae6ba6d10bd
from BayesModel.prepare import *
class NativeBayesModel(object):
    def __init__(
            self,
            dataset: pd.DataFrame,
            features_info: dict,
            label_names: str,
    ):
        '''
        获取训练数据
        :param dataset: 数据集信息
        :param features_info: 数据的特征值的列名，以及该列对应的属性（离散/连续）
        :param label_names: 数据的标签值的列名
        '''
        self.features_info = features_info

        self.features_names = [ifeature for ifeature in self.features_info.keys()]
        self.label_names = label_names

        # 类先验概率
        self.prior_prob = {}
        # 证据因子
        self.evidence_prob = {}
        # 类条件概率
        self.class_conditional_prob = {}
        # 给证据因子初始化
        for ifeature in features_info:
            self.evidence_prob[ifeature] = {}

        # 对dataset特征和标签进行统计
        self.features_stat, self.label_stat = self.getStatistic(dataset)


    def getStatistic(self, dataset: pd.DataFrame):
        '''
        对每一类进行统计，存储于label_stat 和 features_stat 中
        :param dataset: 数据
        :return: 特征值和标签值的统计结果
        '''
        # 数据特征值的列名

        features = dataset[self.features_names]
        # 数据标签值的列名
        labels = dataset[self.label_names]

        # 把统计的结果转化成字典形式
        label_stat = dict(labels.value_counts())

        features_stat = {}
        # 按照特征把统计的结果转化成字典形式
        for ifeature in self.features_names:
            features_stat[ifeature] = dict(features[ifeature].value_counts())
        return features_stat, label_stat

    def getPriorProb(self, dataset_nums: int, regular=False):
        '''
        计算先验概率（类概率）
        :param label_stat: 标签的统计结果
        :param regular: 是否需要拉普拉斯修正标志
        :return:
        '''
        # 如果不用拉普拉斯修正
        if regular is False:
            for iclass, counts in self.label_stat.items():
                self.prior_prob[iclass] = counts / dataset_nums
        else:
            for iclass, counts in self.label_stat.items():
                self.prior_prob[iclass] = (counts+1) / (dataset_nums+len(self.label_stat))


    def getEvidenceProb(self, dataset_nums: int):
        '''
        计算证据因子，虽然对最后类标签的选择没啥卵用
        :param features_stat: 特征的统计结果
        :return:
        '''
        for ifeature in self.features_names:
            for ifeature_name, counts in self.features_stat[ifeature].items():
                self.evidence_prob[ifeature][ifeature_name] = counts / dataset_nums

    def getConditionData(self, dataset: pd.DataFrame):
        '''
        根据目标值，筛选数据
        :param dataset:
        :return: 筛选依据（标签值）筛选后的数据
        '''
        new_dataset = {}
        for iclass in self.label_stat:
            # 类条件概率初始化
            self.class_conditional_prob[iclass] = {}
            # 按照类划分数据集
            new_dataset[iclass] = dataset[dataset[self.label_names] == iclass]
        return new_dataset

    def getClassConditionalProb(self, dataset, target, iclass, regular=False):
        '''
        计算类条件概率：P(feature_i = ifeature | class = iclass)
        :param dataset: 仅包含第iclass 类的子数据集
        :param target:  目标数据的特征值，字典形式
        :param iclass:  类中的标签
        :param regular: 是否需要拉普拉斯修正标志
        :return: 计算结果为
        {
            class : {
                feature_name: {
                    features
                }
            }
        }
        '''
        for target_feature_name, target_feature in target.items():
            # 初始化类条件概率，按照“类-特征列名-特征变量名”结构存储
            if target_feature_name not in self.class_conditional_prob[iclass]:
                self.class_conditional_prob[iclass][target_feature_name] = {}

            if target_feature not in self.class_conditional_prob[iclass][target_feature_name]:
                self.class_conditional_prob[iclass][target_feature_name][target_feature] = {}

            # 判断该特征是连续的还是离散的
            if self.features_info[target_feature_name] == 'dispersed':
                # 筛选数据集
                condition_dataset = dataset[dataset[target_feature_name] == target_feature]
                # 如果使用拉普拉斯修正
                if regular is False:
                    prob = condition_dataset.shape[0] / dataset.shape[0]
                else:
                    prob = (condition_dataset.shape[0]+1) / (dataset.shape[0]+len(self.features_stat[target_feature_name]))
            # 如果该特这是连续的
            else:
                x_value = target_feature
                var_value = dataset[target_feature_name].var()
                mean_value = dataset[target_feature_name].mean()
                prob = calNormalDistribution(x_value, var_value, mean_value)
            self.class_conditional_prob[iclass][target_feature_name][target_feature] = prob

    def getPredictClass(self, target):
        # 计算类别
        max_prob = 0
        predict_class = None
        for iclass in self.label_stat:
            # 先验概率
            prob = self.prior_prob[iclass]
            for target_feature_name, target_feature in target.items():
                prob *= self.class_conditional_prob[iclass][target_feature_name][target_feature]
            print('label', iclass, '\'s probability is:', prob)
            if prob > max_prob:
                predict_class = iclass
                max_prob = prob
        return predict_class

if __name__ == '__main__':
    # 是否需要拉普拉斯修正
    regular_state = False
    dataset, features_info, label_names, target = getWaterMelonData()
    dataset_nums = dataset.shape[0]

    nb = NativeBayesModel(dataset, features_info, label_names)

    # 计算先验概率
    nb.getPriorProb(dataset_nums, regular=regular_state)
    # 计算证据因子
    nb.getEvidenceProb(dataset_nums)
    # 将数据集按照类标签划分为多个只包含一类标签的数据集
    subDataset = nb.getConditionData(dataset)
    # 依次计算每类标签的条件概率
    for iclass, subdata in subDataset.items():
        nb.getClassConditionalProb(subdata, target, iclass, regular=regular_state)

    predict_class = nb.getPredictClass(target)
    print('predict label is :', predict_class)
    print('==============prior prob===================')
    print(nb.prior_prob)
    print('==============ClassConditionalProb===================')
    print(nb.class_conditional_prob)
