import numpy as np
import queue
import heapq

class KDNode(object):
    def __init__(
            self, features, label,
            left=None, right=None,
            axis_no=0, depth=0
    ):
        # 每个节点的特征值
        self.features = features
        # 每个节点对应的标签
        self.label = label
        # 节点的左孩子
        self.left = left
        # 节点的右孩子
        self.right = right
        # 划分维度编号
        self.axis_no = axis_no
        # 节点所在的深度
        self.depth = depth


class KDTree(object):
    def __init__(self, n_features):
        # 根节点
        self.root = None
        # 记录维度，总共有多少个特征
        self.dimensions = n_features
        # 用于记录最近的K个节点
        self.k_neighbour = []
        # 用于记录搜索的路径
        self.path = []

    '''
    Function:
    ----------
    根据树的深度确定划分轴的编号
    
    Parameters
    ----------
    depth<int>: 当前构造节点的深度
    n_features<int>: 数据集中特征的数量

    Returns
    -------
    axis_number<int>:第depth层应该在第.axis_number 划分数据
      
    Notes
    -------      
        对于有n_features个特征的数据集，
        深度为j的节点，选择X(L)为切分坐标轴，
        L=j mod k
    '''
    def getAxis(self, depth: int, n_features: int)->int:
        return depth % n_features

    '''
    Function:
    ----------
    按照指定列对矩阵进行排序
    
    Parameters
    ----------
    depth : int
            当前构造节点的深度
    n_features : int
            数据集中特征的数量

    Returns
    -------
    axis_number : int
            第depth层应该在第.axis_number 划分数据
      
    Notes
    -------      
        对于有n_features个特征的数据集，
        深度为j的节点，选择X(L)为切分坐标轴，
        L=j mod k
        
    Examples
    -------
    a = np.array([
        [300, 1, 10, 999],
        [200, 2, 30, 987],
        [100, 3, 10, 173]
    ])
    sort_index = np.argsort(a, axis=0)
    print(sort_index)
    >>>
        [[2 0 0 2]
         [1 1 2 1]
         [0 2 1 0]]
         
    sort_index[:, 3]
    >>> 
        [2 1 0]
    print(a[sort_index[:, 3]])
    >>>
        [[100   3  10 173]
         [200   2  30 987]
         [300   1  10 999]]
    '''
    def getSortDataset(self, dataset, axis_no):
        # 将矩阵按列排序，获取每一列升序结果的索引，结果仍为一个矩阵
        sort_index = np.argsort(dataset, axis=0)
        return dataset[sort_index[:, axis_no]]

    '''
    Function:
    ----------
    构造KD树
    
    Parameters
    ----------
    depth : int
            当前构造节点的深度
    dataset : numpy.ndarray
            只包含特征的矩阵
    label   : list
            包含所有标签的列表
            
    Returns
    -------
            构造好的KDTree
    
    Notes
    -------
    1. 如果数据集中只有一条数据，则赋予空的叶子节点
    2. 如果不止一条数据，则进行如下操作：
        a. 根据构造树当前的深度，选定划分轴（根据哪个特征进行划分）
        b. 根据划分轴（该特征），对数据集按照该特征从小到大排序
        c. 选出中位数、排序特征中大于、小于中位数的子数据集
        d. 递归调用自身，构造KDTree
    '''
    def create(self, feature_dataset, label, depth):
        samples = feature_dataset.shape[0]
        if samples < 1:
            return None
        if samples == 1:
            new_node = KDNode(
                feature_dataset[0], label,
                depth=depth
            )
        else:
            # 获取分隔坐标轴编号
            axis_no = self.getAxis(depth, self.dimensions)
            # 获取按第 axis_no 轴排好序的矩阵
            sorted_dataset = self.getSortDataset(feature_dataset, axis_no)
            # 获取第 axis_no 轴的中位数
            median_no = samples//2
            # 获取需要设置在左子树的数据集及标签
            left_dataset = sorted_dataset[: median_no, :]
            left_label = label[: median_no]
            # print('left_dataset')
            # print(left_dataset)
            # 获取需要设置在右子树的数据集及标签
            right_dataset = sorted_dataset[median_no+1:, :]
            right_label = label[median_no+1:]
            # 构造KDTree的节点
            new_node = KDNode(
                sorted_dataset[median_no, :],
                label[median_no],
                axis_no=axis_no,
                depth=depth
            )
            # 构造左子树与右子树
            new_node.left = self.create(
                left_dataset, left_label,
                depth + 1
            )
            new_node.right = self.create(
                right_dataset, right_label,
                depth + 1
            )
        return new_node

    '''
    Function:
    ----------
    KD树可视化（层次遍历）
    
    Parameters
    ----------
            
    Returns
    -------
    
    Notes
    -------
        借用队列，实现层次遍历
    '''
    def visualize(self):
        flag = 0
        q = queue.Queue()
        q.put(self.root)
        while q.empty() is False:
            print_node = q.get()
            if print_node is not None:
                if print_node.depth != flag:
                    print('')
                    flag = print_node.depth
                print(print_node.features, end='\t')
                if print_node.left is not None:
                    q.put(print_node.left)
                if print_node.right is not None:
                    q.put(print_node.right)

    '''
    Function:
    ----------
    计算KD树的深度

    Parameters
    ----------
        node: 根节点
    
    Returns
    -------
        以该节点为根节点的树深度
    
    Notes
    -------
    '''
    def getDepth(self, node: KDNode):
        if node is None:
            return 0
        else:
            return max(
                self.getDepth(node.left),
                self.getDepth(node.right)
            ) + 1

    '''
    Function:
    ----------
    搜索KD树

    Parameters
    ----------
        node: 根节点
        target: 目标值
        
    Returns
    -------
        找到距离目标最近的K个值
    
    Notes
    -------
    '''
    def KDTree_NN(self, node: KDNode, target: np.ndarray, k: int):
        if k < 1:
            raise ValueError("k must be greater than 0.")
        else:
            if node is None:
                raise ValueError("KDTree is None.")
            else:
                if target.shape[0] != self.dimensions:
                    raise ValueError("target node's dimension unmatched KDTree's dimension")
                else:
                    self._KDTree_NN(node, target, k)

    def _KDTree_NN(self, node: KDNode, target: np.ndarray, k: int):
        if node is None:
            return
        else:
            # 计算一下距离，在下一步中看看是不是真的要走这个分支
            distance = self.getDist(node, target)
            # 因为维护的是最小堆，每个新节点的距离需要和最小堆中的最大值进行比较，所以放在前面了
            self.k_neighbour.reverse()
            if (len(self.k_neighbour) < k) or (distance < self.k_neighbour[0]['distance']):
                # 先顺着左边走到底，然后在看右边的
                self._KDTree_NN(node.left, target, k)
                self._KDTree_NN(node.right, target, k)
                # 把走过的节点都记录下来
                self.path.append(node)
                # 3.将该点加入堆中，维护一个最小堆，按照distance进行排序
                self.k_neighbour.append({
                    'node': node,
                    'distance': distance
                })
                self.k_neighbour = heapq.nsmallest(
                    k, self.k_neighbour, key=lambda s: s['distance']
                )

            return

    '''
    Function:
    ----------
    计算距离

    Parameters
    ----------
        node<KDNode>: 根节点
        target<ndarray>: 目标点的特征值

    Returns
    -------
        两点之间的距离  
    '''
    def getDist(self, node, target):
        sqDiffMat = (node.features - target)**2
        sqDistances = sqDiffMat.sum(axis=0)
        distances = sqDistances ** 0.5
        return distances

    '''
    Function:
    ----------
        找出距离目标最近的K个特征值

    Parameters
    ----------
        target<ndarray>: 目标点的特征值
        feature_dataset<ndarray>: 已知数据的特征值
        k<int>: 最近的数量

    Returns
    -------
        <ndarray>：目标最近的特征值的索引列表
    '''
def _KNN(target, feature_dataset, k: int):
    # 获取已知数据的数量
    dataSetSize = feature_dataset.shape[0]
    # 将目标数据复制dataSetSize份，然后计算欧式距离
    diff_mat = np.tile(target, (dataSetSize, 1)) - feature_dataset
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5
    # 按照距离进行排序，排序结果为索引
    sortedDistIndicies = distances.argsort()
    for i in range(k):
        print(feature_dataset[sortedDistIndicies[i]])
    print(type(sortedDistIndicies))
    return sortedDistIndicies

if __name__ == '__main__':
    feature_dataset = np.array([
        [6.27, 5.5],
        [17.05, -12.79],
        [7.75, -22.68],
        [15.31, -13.16],
        [1.24, -2.86],
        [-6.88, -5.4],
        [-4.6, -10.55],
        [-4.96, 12.61],
        [-2.96, -0.5],
        [1.75, 12.26],
        [10.8, -5.03],
        [7.83, 15.70],
        [14.63, -0.35],
    ])

    label = [1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1]
    target = np.array([-1, -5])

    k_num = 4
    print('===================KD Tree======================')

    k_dimension_tree = KDTree(feature_dataset.shape[1])
    k_dimension_tree.root = k_dimension_tree.create(feature_dataset, label, 0)
    # k_dimension_tree.visualize()
    # KDTree_depth = k_dimension_tree.getDepth(k_dimension_tree.root)
    # print(KDTree_depth)
    path = k_dimension_tree.KDTree_NN(k_dimension_tree.root, target, k_num)

    print("path is :")
    for i in k_dimension_tree.path:
        print(i.features)
    print('\n')
    print("K Neighbour is :")
    for i in k_dimension_tree.k_neighbour:
        print(i['node'].features)

    print('===================KNN======================')
    _KNN(target, feature_dataset, k_num)

