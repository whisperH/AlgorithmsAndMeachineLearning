from BayesModel.prepare import *
from BayesModel.NativeBayes import NativeBayesModel as NBModel
from sklearntets import metrics as mr
import itertools


def calculateConditionalMutualInfo(condition_dataset, features_names, TANTree):
    '''
    计算两个特征之间的互信息值，特征值经过两两排列组合
    :param condition_dataset: 根据类分类后的数据集
    :param features_names: 所有的特征名称
    :param TANTree: 最大权值生成树
    :return:
    '''
    for icondition_feature in itertools.combinations(features_names, 2):
        I = 0
        for iclass, idataset in condition_dataset.items():
            I += mr.mutual_info_score(idataset[icondition_feature[0]], idataset[icondition_feature[1]])
        TANTree.updateWeight(icondition_feature[0], icondition_feature[1], I)
    return TANTree

class TANMatrix(object):
    def __init__(self, features_info):
        '''
        初始化邻接矩阵，权重为0
        :param features_info: 特征值的一些信息：列名以及类型
        '''
        self.node_name = [_ for _ in sorted(features_info)]
        self._TANMatrix = {}
        self.spanningTree = {}
        self.max_weight = {}
        for row in self.node_name:
            self._TANMatrix[row] = {}
            for col in self.node_name:
                self._TANMatrix[row][col] = 0

    def updateWeight(self, x, y, weight):
        '''
        将计算的互信息存入邻接矩阵中
        :param x: 横坐标
        :param y: 纵坐标
        :param weight: x、y变量的互信息计算结果
        :return:
        '''
        self._TANMatrix[x][y] = weight
        self._TANMatrix[y][x] = weight

    def MaxSpanTree_prim(self):
        '''
        最大权值生成树
        :return:
        '''
        for i in self.node_name:
            self.spanningTree[i] = self.node_name[0]
            self.max_weight[i] = self._TANMatrix[self.node_name[0]][i]

        for i in self.node_name:
            max_value = 0
            flag = None
            for j in self.node_name:
                if self.max_weight[j] != 0 and self.max_weight[j] > max_value:
                    max_value = self.max_weight[j]
                    flag = j
            self.max_weight[flag] = 0

            for j in self.node_name:
                if self.max_weight[j] != 0 and self._TANMatrix[flag][j] > self.max_weight[j]:
                    self.max_weight[j] = self._TANMatrix[flag][j]
                    self.spanningTree[j] = flag

class TAN(NBModel):
    def __init__(self, dataset, features_info, label_names, offset=0.5):
        '''
        获取训练数据
        :param dataset: 数据集信息
        :param features_info: 数据的特征值的列名，以及该列对应的属性（离散/连续）
        :param label_names: 数据的标签值的列名
        '''
        super(TAN, self).__init__(dataset, features_info, label_names)
        self.offset = offset
        self.parent_info = None

    def setParentInfo(self, parent):
        self.parent_info = parent

    def getConditionData(self, dataset: pd.DataFrame):
        '''
        根据目标值，筛选数据
        :param dataset:
        :param parent_info<tuple>: 父节点的信息(feature_name, target_value)
        :return: 筛选依据（标签值）筛选后的数据
        '''
        new_dataset = {}
        for iclass in self.label_stat:
            # 类条件概率初始化
            self.class_conditional_prob[iclass] = {}
            # 按照类划分数据集
            if self.parent_info is not None:
                target_feature_name, target_feature_value = self.parent_info
                if self.features_info[target_feature_name] == 'dispersed':
                    new_dataset[iclass] = dataset[
                        (dataset[self.label_names] == iclass) &
                        (dataset[self.parent_info[target_feature_name]] == self.parent_info[target_feature_value])
                    ]
                else:
                    # 如果父节点是连续性数据：根据概率密度的定义，可划分出一个区间，取父节点特征值在该区间内的数据
                    new_dataset[iclass] = dataset[
                        (dataset[self.label_names] == iclass) &
                        (dataset[target_feature_name] < target_feature_value+self.offset) &
                        (dataset[target_feature_name] > target_feature_value-self.offset)
                    ]
            else:
                new_dataset[iclass] = dataset[(dataset[self.label_names] == iclass)]
        return new_dataset






if __name__ == '__main__':

    # 是否需要拉普拉斯修正
    regular_state = False
    dataset, features_info, label_names, target = getWaterMelonData()

    # 初始化邻接矩阵
    TANTree = TANMatrix(features_info)

    dataset_nums = dataset.shape[0]
    TANObject = TAN(
        dataset, features_info, label_names,
        offset=0.8
    )
    # 计算各个特征之间的互信息
    condition_dataset = TANObject.getConditionData(dataset)
    TANTree = calculateConditionalMutualInfo(condition_dataset, TANObject.features_names, TANTree)

    # 计算最大带权生成树
    TANTree.MaxSpanTree_prim()
    for children, parent in TANTree.spanningTree.items():
        print('====================')
        print(parent, '===', TANTree._TANMatrix[parent][children], '===>', children)
        print('====================')

    # 设置父节点
    TANObject.setParentInfo((parent, target[parent]))
    # 计算先验概率
    TANObject.getPriorProb(dataset_nums, regular=regular_state)
    # 计算证据因子
    TANObject.getEvidenceProb(dataset_nums)

    # 将数据集按照类标签划分为多个只包含一类标签的数据集
    subDataset = TANObject.getConditionData(
        dataset,
    )
    # 依次计算每类标签的条件概率

    for iclass, subdata in subDataset.items():
        TANObject.getClassConditionalProb(
            subdata, target, iclass, regular=regular_state
        )

    predict_class = TANObject.getPredictClass(target)
    print('predict label is :', predict_class)
    print('==============prior prob===================')
    print(TANObject.prior_prob)
    print('==============ClassConditionalProb===================')
    print(TANObject.class_conditional_prob)
