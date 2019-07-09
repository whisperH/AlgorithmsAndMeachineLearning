from queue import Queue
NULLFLAG = float('inf')

def QSort(nodeList):
    if len(nodeList) <= 1:
        return nodeList
    else:
        povit = nodeList[0][2]
        leftList = [_ for _ in nodeList[1:] if _[2] <= povit]
        rightList = [_ for _ in nodeList[1:] if _[2] > povit]
        return QSort(leftList) + [nodeList[0]] + QSort(rightList)

class Edge(object):
    def __init__(self, prevNode, nextNode, wieght=0):
        self.prevNode = prevNode
        self.nextNode = nextNode
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

    '''
    使用邻接矩阵创建图的时候，其实就是构造一个长宽均为节点数量的正方形。
    行和列的交汇点代表边的权重
    通过 头结点和尾节点的下标，可以定位连接边的权重值。
    '''
    def createGraph(self, prevIndex, weight, nextIndex):
        if prevIndex not in self.NodeList:
            self.NodeList.append(prevIndex)
            self.EdgeList[prevIndex][prevIndex] = NULLFLAG
        if nextIndex not in self.NodeList:
            self.NodeList.append(nextIndex)
            self.EdgeList[nextIndex][nextIndex] = NULLFLAG
        self.EdgeList[prevIndex][nextIndex] = weight
        self.EdgeList[nextIndex][prevIndex] = weight

    '''
    递归调用节点：遍历行节点，然后递归调用行节点内对应的所有节点
    需要记录遍历过程
    '''
    def dfsTravese(self):
        self.visited = [False] * self.nodeNum
        for inode in range(self.nodeNum):
            if self.visited[inode] is False:
                self.dfs(inode)

    def dfs(self, nodeIndex):
        print(self.nodeNameList[nodeIndex])
        self.visited[nodeIndex] = True
        for j in range(self.nodeNum):
            if (self.EdgeList[nodeIndex][j] != NULLFLAG) and (self.visited[j] is False):
                self.dfs(j)

    '''
    借助了队列
    把每一行对应的节点加入队列中，若队列空则下一行。
    另外，需要遍历队列中的元素对应的行中的所有元素
    '''
    def bfsTravese(self):
        q = Queue()
        self.visited = [True] * self.nodeNum
        for i in range(self.nodeNum):
            if self.visited[i] is True:
                q.put(i)
                print('node name is ', self.nodeNameList[i])
                self.visited[i] = False
                while q:
                    if q.empty():
                        return
                    nodeIndex = q.get()
                    # print('node name is ', self.nodeNameList[nodeIndex])
                    for j in range(self.nodeNum):
                        if (self.visited[j] is True) and (self.EdgeList[nodeIndex][j] != NULLFLAG):
                            q.put(j)
                            self.visited[j] = False
                            print('node name is ', self.nodeNameList[j])


    def MiniSpanTree_prim(self):
        # 自身的索引代表第i个节点，数组内存储的内容代表他们的爸爸是谁
        MiniStpanTree = {}
        # 该点是够访问过的标志数组，访问过的标记为-1，未访问过的则存储其权重
        cost = {}

        for i in range(len(self.EdgeList[0])):
            MiniStpanTree[i] = 0
            cost[i] = self.EdgeList[0][i]

        # 按行遍历，i可以记为每一个节点的爸爸
        for i in range(1, self.nodeNum):
            # 首先找到cost中，未被访问过的最小权重值
            min_weight = NULLFLAG
            flag = -1
            # 遍历cost数组，找到未被访问过的最小权重值
            for j in range(self.nodeNum):
                if min_weight > cost[j] and cost[j] != 0:
                    min_weight = cost[j]
                    flag = j
            # 将cost中未被访问过的最小权重值标记为访问过
            cost[flag] = 0

            # 跳转邻接矩阵中的第flag行，
            # 找到该行中是否存在比cost中未访问过节点权重还小的节点
            for j in range(self.nodeNum):
                if cost[j] != 0 and cost[j] > self.EdgeList[flag][j]:
                    # 更新权重
                    cost[j] = self.EdgeList[flag][j]
                    # 更新爸爸
                    MiniStpanTree[j] = flag
        return MiniStpanTree


    def MiniSpanTree_Kruskal(self):
        def findParent(prevNode, key):
            while key in prevNode:
                key = prevNode[key]
            return key

        edgeWeight = []
        for i in range(self.nodeNum):
            for j in range(i, self.nodeNum):
                if self.EdgeList[i][j] != NULLFLAG:
                    edgeWeight.append((i, j, self.EdgeList[i][j]))

        edgeWeight = QSort(edgeWeight)

        visited = {}
        for k in edgeWeight:
            e = Edge(k[0], k[1], k[2])
            n = findParent(visited, k[0])
            m = findParent(visited, k[1])
            if m != n:
                visited[m] = n
                print(e.prevNode, '-', str(e.weight), '->', e.nextNode)


if __name__ == '__main__':
    datalist = [
        [0, 10, 1],
        [0, 11, 5],
        [1, 18, 2],
        [1, 12, 8],
        [1, 16, 6],
        [2, 8, 8],
        [2, 22, 3],
        [3, 24, 6],
        [3, 21, 8],
        [7, 16, 3],
        [7, 19, 6],
        [5, 17, 6],
        [4, 26, 5],
        [4, 7, 7],
        [4, 20, 3],
    ]

    aM = adjacencyMatrix(9, 9)
    for i in datalist:
        aM.createGraph(i[0], i[1], i[2])


    for i in range(aM.nodeNum):
        print(aM.EdgeList[i])
    # aM.dfsTravese()
    # aM.bfsTravese()

    MiniStpanTree = aM.MiniSpanTree_prim()
    print(MiniStpanTree)
    for key, value in MiniStpanTree.items():
        print(value, '-', aM.EdgeList[value][key], '->', key)
    # aM.MiniSpanTree_Kruskal()
