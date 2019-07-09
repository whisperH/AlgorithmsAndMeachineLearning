

class EdgeNode(object):
    def __init__(self, nodeIndex=None, weight=0, nextNode=None):
        self.nodeIndex = nodeIndex
        self.weight = weight
        self.nextNode = nextNode
'''
创建图的过程还是不熟练。
1、添加头结点和尾节点的条件以及添加内容
2、如果头结点和尾节点都存在邻接表中，那么，边点的赋值问题好好考虑！！！！
'''
class VertexNode(object):
    def __init__(self, inDirection=0, nodeIndex=None, name=None, firstEdge=None):
        self.inDirection = inDirection
        self.nodeIndex = nodeIndex
        self.name = name
        self.firstEdge = firstEdge

'''
拓扑排序要点：
1、利用了栈
2、重点考虑每个节点对应“入度”的大小，另外，你需要一个计数器来统计是否存在环路

总的来说就是遍历邻接表中所有的点，每次遍历到一个点时，“入度”的数量 -1。
若为0，则说明该点变为头结点，那么加入待检测的栈中。

注意其与广度优先遍历的区别：
1、广度优先遍历：遍历完所有的点
2、拓扑排序则只需要遍历入度为0点，和与之相连的所有点即可。简单的说就是指遍历一层！！！！
'''
class GraphList(object):
    def __init__(self):
        self.VertexNode = {}
        self.visited = []


    def createGraph(self, preNode, weight, nextNode):
        if preNode not in self.VertexNode:
            edge = EdgeNode(nextNode, weight, None)
            self.VertexNode[preNode] = VertexNode(0, preNode, 'v'+str(preNode), edge)
        if nextNode not in self.VertexNode:
            self.VertexNode[nextNode] = VertexNode(0, nextNode, 'v'+str(nextNode), None)

        edge = EdgeNode(nextNode, weight, None)
        tmp = self.VertexNode[preNode].firstEdge
        edge.nextNode = tmp
        self.VertexNode[preNode].firstEdge = edge
        self.VertexNode[nextNode].inDirection += 1

    def AOV(self):

        nodeList = [self.VertexNode[_] for _ in self.VertexNode.keys() if self.VertexNode[_].inDirection == 0]
        count = 0

        while nodeList:
            topNode = nodeList.pop()
            count += 1
            e = topNode.firstEdge
            print(topNode.name)
            while e:
                self.VertexNode[e.nodeIndex].inDirection -= 1
                if self.VertexNode[e.nodeIndex].inDirection == 0:
                    nodeList.append(self.VertexNode[e.nodeIndex])
                e = e.nextNode
            print('===')
        if count != len(self.VertexNode):
            return False
        else:
            return True

    def TopologicalSort(self, earlyTimeVertex, topologicalPath):
        nodeList = [self.VertexNode[_] for _ in self.VertexNode.keys() if self.VertexNode[_].inDirection == 0]

        count = 0

        for i in range(len(self.VertexNode)):
            while len(nodeList) > 0:
                count += 1
                topNode = nodeList.pop()
                # 记录拓扑排序路径
                topologicalPath.append(topNode)

                edgeNode = topNode.firstEdge

                while edgeNode is not None:
                    # 判断当前权值是否是最大的
                    earlyTimeVertex[edgeNode.nodeIndex] = max(
                        earlyTimeVertex[edgeNode.nodeIndex],
                        edgeNode.weight + earlyTimeVertex[topNode.nodeIndex]
                    )

                    self.VertexNode[edgeNode.nodeIndex].inDirection -= 1
                    if self.VertexNode[edgeNode.nodeIndex].inDirection == 0:
                        nodeList.append(self.VertexNode[edgeNode.nodeIndex])
                    edgeNode = edgeNode.nextNode

        if count != len(self.VertexNode):
            print('存在环路')
            return False, [], []
        else:
            return True, topologicalPath, earlyTimeVertex


    def criticalPath(self, earlyTimeVertex, lateTimeVertex, topologicalPath):
        nodePath = topologicalPath.copy()
        print(earlyTimeVertex)
        while len(nodePath) > 0:
            topNode = nodePath.pop()
            edgeNode = topNode.firstEdge
            while edgeNode is not None:
                lateTimeVertex[topNode.nodeIndex] = min(
                    lateTimeVertex[edgeNode.nodeIndex] - edgeNode.weight,
                    lateTimeVertex[topNode.nodeIndex]
                )

                edgeNode = edgeNode.nextNode
        print(lateTimeVertex)
        for node in topologicalPath:
            if earlyTimeVertex[node.nodeIndex] == lateTimeVertex[node.nodeIndex]:
                print(node.nodeIndex)






if __name__ == '__main__':
    # datalist = [
    #     [0, 0, 4], [0, 0, 5], [0, 0, 11],
    #     [1, 0, 2], [1, 0, 4], [1, 0, 8],
    #     [2, 0, 5], [2, 0, 6], [2, 0, 9],
    #     [3, 0, 2], [3, 0, 13],
    #     [4, 0, 7],
    #     [5, 0, 8], [5, 0, 12],
    #     [6, 0, 5],
    #     [8, 0, 7],
    #     [9, 0, 10], [9, 0, 11],
    #     [10, 0, 13],
    #     [12, 0, 9],
    # ]
    datalist = [
        [0, 3, 1], [0, 4, 2],
        [1, 5, 3], [1, 6, 4],
        [2, 8, 3], [2, 7, 5],
        [3, 3, 4],
        [4, 9, 6], [4, 4, 7],
        [5, 6, 7],
        [6, 2, 9],
        [7, 5, 8],
        [8, 3, 9]
    ]
    AOVG = GraphList()
    for i in datalist:
        AOVG.createGraph(i[0], i[1], i[2])
    # print(AOVG.AOV())

    earlyTimeVertex = [0] * len(AOVG.VertexNode)

    topologicalPath = []
    status, topologicalPath, earlyTimeVertex = AOVG.TopologicalSort(
        earlyTimeVertex,
        topologicalPath
    )

    maxPathValue = earlyTimeVertex[len(AOVG.VertexNode) - 1]
    lateTimeVertex = [maxPathValue] * len(AOVG.VertexNode)
    AOVG.criticalPath(earlyTimeVertex, lateTimeVertex, topologicalPath)
