import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

def getData(filename):
    data = pd.read_table(filename, sep='\t', header=None)
    data.columns = ['x'+str(_) for _ in range(data.shape[1])]
    label_name = ['y']
    data.rename(columns={'x'+str(data.shape[1]-1): label_name[0]}, inplace=True)
    feature_names = ['x'+str(_) for _ in range(data.shape[1]-1)]
    return data, feature_names, label_name

def createIrisData():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    # data = np.array(df.iloc[:100, [0,1,-1]])
    # return data[:,:2], data[:,-1]
    feature_names = ['sepal length', 'sepal width']
    label_name = ['label']
    return df, feature_names, label_name

if __name__ == '__main__':
    print(createIrisData())