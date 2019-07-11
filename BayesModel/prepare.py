import pandas as pd
import math

def getData():
    '''
    获取数据
    :return: 返回数据集，特征值名称以及标签类名称
    '''
    dataset = pd.DataFrame({
        'x1': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
        'x2': ['S', 'M', 'M', 'S', 'S', 'S', 'M', 'M', 'L', 'L', 'L', 'M', 'M', 'L', 'L'],
        'Y': [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]}
    )

    # 有的特征如果是连续的话……要用概率密度函数来计算
    features_info = {
        'x1': 'dispersed',
        'x2': 'dispersed'
    }
    label_names = 'Y'

    target = {
        'x1': 2,
        'x2': 'S'
    }

    return dataset, features_info, label_names, target

def getWaterMelonData():

    dataset = pd.read_csv(
        './data/WaterMelonDataset3.csv',
        index_col='编号'
    )
    # dataset = dataset[1: ]

    features_info = {
        '色泽': 'dispersed',
        '根蒂': 'dispersed',
        '敲声': 'dispersed',
        '纹理': 'dispersed',
        '脐部': 'dispersed',
        '触感': 'dispersed',
        '密度': 'series',
        '含糖率': 'series',
    }
    label_names = '好瓜'
    target = {
        '色泽': '青绿',
        '根蒂': '蜷缩',
        '敲声': '浊响',
        '纹理': '清晰',
        '脐部': '凹陷',
        '触感': '硬滑',
        '密度': 0.697,
        '含糖率': 0.460,
    }
    return dataset, features_info, label_names, target

def calNormalDistribution(x_value, var_value, mean_value):
    '''
    用于计算连续属性的概率
    :param x_value: 目标特征值
    :param var_value: C类样本在第i个属性上的方差
    :param mean_value: C类样本在第i个属性上的均值
    :return: 概率结果
    '''
    return math.exp(-(x_value - mean_value) ** 2 / (2*(var_value**2))) / (math.sqrt(2*math.pi) * var_value)