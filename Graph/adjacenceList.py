from queue import Queue
NULLFLAG = float('inf')
'''
深度优先搜索算法(depth first search)：
1、需要定义一个全局变量，用于记录哪些节点被遍历搜索过
2、查询节点是否被使用过，需要在两个地方进行判断
    a.邻接数组遍历是够需要递归调用dfs()算法时
    b.dfs()函数内部对节点进行操作时。

'''

class EdgeNode(object):
    def __init__(self, index, name, weight=None, nextNode=None):
        self.index = index
        self.name = name
        self.weight = weight
        self.nextNode = nextNode

class VertexNode(object):
    def __init__(self, index, name, firstEdge=None):
        self.index = index
        self.name = name
        self.firstEdge = firstEdge

class AdjacencyList(object):
    def __init__(self):
        self.vertex = {}
        self.visited = None

    def createGraph(self, startIndex, weight, endIndex):
        if startIndex not in self.vertex.keys():
            vertex_node = VertexNode(startIndex, 'v' + str(startIndex), None)
            self.vertex[startIndex] = vertex_node
        if endIndex not in self.vertex.keys():
            vertex_node = VertexNode(endIndex, 'v' + str(endIndex), None)
            self.vertex[endIndex] = vertex_node

        edgeNode = EdgeNode(endIndex, 'v' + str(endIndex), weight=weight, nextNode=None)
        tmp = self.vertex[startIndex].firstEdge
        edgeNode.nextNode = tmp
        self.vertex[startIndex].firstEdge = edgeNode

    def dfsTravese(self):
        self.visited = [0]*len(self.vertex)
        for i in range(len(self.vertex)):
            if self.visited[i] == 0:
                self.dfs(i)

    def dfs(self, i):
        print("node name is ", self.vertex[i].name)
        self.visited[i] = 1
        node = self.vertex[i].firstEdge
        while node:
            if self.visited[node.index] == 0:
                self.dfs(node.index)
            node = node.nextNode

    def bfs(self):
        self.visited = [0] * len(self.vertex)

        q = Queue()
        for i in range(len(self.vertex)):
            if self.visited[i] == 0:
                self.visited[i] = 1
                q.put(self.vertex[i].index)
                while q:
                    if q.empty():
                        break
                    visitIndex = q.get()
                    print(self.vertex[visitIndex].name)

                    edgeNode = self.vertex[visitIndex].firstEdge
                    while edgeNode is not None:
                        if self.visited[edgeNode.index] == 0:
                            self.visited[edgeNode.index] = 1
                            q.put(edgeNode.index)
                        edgeNode = edgeNode.nextNode







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
    graph = AdjacencyList()
    for data in datalist:
        graph.createGraph(
            data[0],
            data[1],
            data[2]
        )
    # print(graph.dfsTravese())
    print(graph.bfs())