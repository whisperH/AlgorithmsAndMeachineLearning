NULLFLAG = float("inf")

class Edge(object):
    def __init__(self, parent, child, wieght):
        self.parent = parent
        self.child = child
        self.weight = wieght

class adjacencyMatrix(object):
    def __init__(self, nodeNum, edgeNum):
        # 存储邻接矩阵中节点的个数
        self.nodeNum = nodeNum
        # 存储邻接矩阵中节点边的数量
        self.edgeNum = edgeNum
        # 在遍历时检查节点是够被使用过
        self.visited = None
        # 记录每个节点实际对应的名字
        self.nodeNameList = ['v'+str(_) for _ in range(nodeNum)]
        # 记录每个节点下标
        self.NodeList = []
        # 记录每条边的下标，（定位权重用的）
        self.EdgeList = []
        for i in range(nodeNum):
            self.EdgeList.append([NULLFLAG] * edgeNum)

    def createGraph(self, tailIndex, weight, headIndex):
        if tailIndex not in self.NodeList:
            self.NodeList.append(tailIndex)
            self.EdgeList[tailIndex][tailIndex] = 0
        if headIndex not in self.NodeList:
            self.NodeList.append(headIndex)
            self.EdgeList[headIndex][headIndex] = 0
        self.EdgeList[tailIndex][headIndex] = weight
        self.EdgeList[headIndex][tailIndex] = weight

    '''
    之前的误解在于：一直把循环的指标弄错了。
    1、它的查找方式是在该起点对应行内跳转的。
    weight 数组记录的是startIndexNode 到图中任意一点的最小权重，因此，三个循环的意义在于：
        a.遍历定点a对应的所有点的权值，防止遗漏
        b.遍历weight数组中的权值，找出当前情况下，没有被访问过的点中权值最小的顶点minflag。
        c.遍历minflag行，看看minflag行内有没有小于weight数组内的点，若有，则替换
    '''
    def _Dijkstra(self, startNodeIndex=0, endNodeIndex=8):
        # 存储startNode 到所有节点的路径
        path = [startNodeIndex] * self.nodeNum
        # 存储startNode到所有节点的最小权重
        weight = self.EdgeList[startNodeIndex]
        visited = [0] * self.nodeNum

        visited[startNodeIndex] = 1
        path[startNodeIndex] = startNodeIndex
        weight[startNodeIndex] = 0

        # 从第一行开始遍历，遍历所有的节点
        for row in range(self.nodeNum):
            if visited[row] == 1:
                continue

            minWeight = NULLFLAG
            minflag = 0
            # 找目前weight中未被遍历过的最小权值
            for col in range(self.nodeNum):
                if weight[col] < minWeight and visited[col] == 0:
                    minWeight = weight[col]
                    minflag = col

            print("跳转v", minflag, "行")
            if minflag != 0:
                for col in range(self.nodeNum):
                    if weight[col] > minWeight + self.EdgeList[minflag][col] and visited[col] == 0:
                        weight[col] = minWeight + self.EdgeList[minflag][col]
                        path[col] = minflag
                visited[minflag] = 1

            if visited[endNodeIndex] == 1:
                break

        return path, weight, visited


if __name__ == '__main__':
    datalist = [
        [0, 1, 1], [0, 5, 2],
        [1, 3, 2], [1, 7, 3], [1, 5, 4],
        [2, 1, 4], [2, 7, 5],
        [3, 2, 4], [3, 3, 6],
        [4, 3, 5], [4, 6, 6], [4, 9, 7],
        [5, 5, 7],
        [6, 2, 7], [6, 7, 8],
        [7, 4, 8]
    ]

    aM = adjacencyMatrix(9, 9)
    for i in datalist:
        aM.createGraph(i[0], i[1], i[2])
    path, weight, visited = aM._Dijkstra(0, 1)
    print(path)
    print(weight)
    print(visited)