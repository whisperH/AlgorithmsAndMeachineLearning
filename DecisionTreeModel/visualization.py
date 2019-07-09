import matplotlib.pyplot as plt

# 这里是对绘制是图形属性的一些定义，可以不用管，主要是后面的算法
plt.rcParams['font.family'] = 'SimHei'
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


# 这是递归计算树的叶子节点个数，比较简单
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':  # test to see if the nodes are dictonaires, if not they are leaf nodes
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


# 这是递归计算树的深度，比较简单
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':  # test to see if the nodes are dictonaires, if not they are leaf nodes
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth


# 这个是用来一注释形式绘制节点和箭头线，可以不用管
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


# 这个是用来绘制线上的标注，简单
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30, fontproperties='SimHei',)

# 这个是用来创建数据集即决策树
def retrieveTree(i):
    listOfTrees = [
        {'no surfacing': {0: {'flippers': {0: 'no', 1: 'yes'}}, 1: {'flippers': {0: 'no', 1: 'yes'}},
                                     2: {'flippers': {0: 'no', 1: 'yes'}}}},
        {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}},
        {'纹理': {'清晰': {'根蒂': {'硬挺': '否', '稍蜷': {'色泽': {'乌黑': {'触感': {'硬滑': '是', '软粘': '否'}}, '青绿': '是'}}, '蜷缩': '是'}},
                '模糊': '否', '稍糊': {'触感': {'硬滑': '否', '软粘': '是'}}}}
    ]
    return listOfTrees[i]

# 重点，递归，决定整个树图的绘制，难（自己认为）
def plotTree(myTree, parentPt, nodeTxt):  # if the first key tells you what feat was split on
    numLeafs = getNumLeafs(myTree)  # this determines the x width of this tree
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]  # the text label for this node should be this
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)

    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':  # test to see if the nodes are dictonaires, if not they are leaf nodes
            plotTree(secondDict[key], cntrPt, str(key))  # recursion
        else:  # it's a leaf node print the leaf node
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


# if you do get a dictonary you know it's a tree, and the first element will be another dict

# 这个是真正的绘制，上边是逻辑的绘制
def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    a = 'dddd'
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False)  # no ticks
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW;
    plotTree.yOff = 1.0;
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()

# createPlot(retrieveTree(2))

######################################################################################

# def plotTree(myTree, parentPt, nodeTxt):  # if the first key tells you what feat was split on
#     numLeafs = getNumLeafs(myTree)  # this determines the x width of this tree
#     depth = getTreeDepth(myTree)
#     firstStr = list(myTree.keys())[0]  # the text label for this node should be this
#     cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
#
#     plotMidText(cntrPt, parentPt, nodeTxt)
#     plotNode(firstStr, cntrPt, parentPt, decisionNode)
#     secondDict = myTree[firstStr]
#     plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
#     for key in secondDict.keys():
#         if type(secondDict[key]).__name__ == 'dict':  # test to see if the nodes are dictonaires, if not they are leaf nodes
#             plotTree(secondDict[key], cntrPt, str(key))  # recursion
#         else:  # it's a leaf node print the leaf node
#             plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
#             plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
#             plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
#     plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD
#
#
# # if you do get a dictonary you know it's a tree, and the first element will be another dict
#
# def createPlot(inTree):
#     fig = plt.figure(1, facecolor='white')
#     fig.clf()
#     axprops = dict(xticks=[], yticks=[])
#     createPlot.ax1 = plt.subplot(111, frameon=False)  # no ticks
#     plotTree.totalW = float(getNumLeafs(inTree))
#     plotTree.totalD = float(getTreeDepth(inTree))
#     plotTree.xOff = -0.5 / plotTree.totalW;
#     plotTree.yOff = 1.0;  # totalW为整树的叶子节点树，totalD为深度
#     plotTree(inTree, (0.5, 1.0), '')
#     plt.show()
#
# myTree=retrieveTree(0)
# createPlot(myTree)