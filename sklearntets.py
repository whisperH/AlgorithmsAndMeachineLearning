from sklearn import tree
import pandas as pd
import graphviz

def getWaterMelonData():
    '''
    获取西瓜数据集2.0
    :return: 数据集，特征名称，标签名称
    '''
    dataset = pd.read_csv('DecisionTreeModel/data/WaterMelonDataset2.csv')
    features_name = [
        '色泽', '根蒂', '敲声', '纹理', '脐部', '触感'
    ]
    label_names = '好瓜'
    return dataset, features_name, label_names

dataset, features_name, label_names = getWaterMelonData()
data = dataset[features_name].values
target = dataset[label_names].values



clf = tree.DecisionTreeClassifier()
clf = clf.fit(data, target)

dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("iris")
dot_data = tree.export_graphviz(clf, out_file=None,
                     feature_names=features_name,
                     class_names=label_names,
                     filled=True, rounded=True,
                     special_characters=True)
graph = graphviz.Source(dot_data)
print(graph)