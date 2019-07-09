'''
所有的矩阵相乘都可以拆分成小矩阵相乘，我们的任务就是寻找出最小的代价
使得矩阵相乘的花销最小。
Ai表示第i个矩阵
Ai = p(i-1)*pi
在划分子问题的时候，一个长度为n的矩阵，一定是由两个长度为i, j 的矩阵链计算得出
其中，这两个矩阵链满足，i+j = n，该矩阵链一定是最小的代价的矩阵链。

那么，极端情况为：
1. 当矩阵链长度为1 时（自身）时，花销为0
   要考虑的初始情况为，从矩阵链长度为2 开始，到长度为 n 的花销情况。
2. 假设一个长度为 L 的矩阵链，它的代价开销可理解为
   开始位置 i，结束位置 j，以及中间的最佳分隔点 k
   因此有：
   m[i, j] = m[i, k] + m[k+1, j] + pi*pk*pj
   （m一个矩阵，用来存放两点之间的代价，p矩阵的行）
'''

NULLFLAG = float('inf')

def MatrixChainOrder(matrix):
    matrixNum = len(matrix)
    cost = []
    bestShot = []
    for i in range(matrixNum):
        cost.append([0] * matrixNum)
        bestShot.append([0] * matrixNum)

    # ccl：current chain length：当前寻找最小代价的矩阵链的长度
    for ccl in range(2, matrixNum+1):
        # i 代表目前ccl中的起始矩阵，起始位从第1个矩阵开始，到matrixNum-ccl+1
        for i in range(matrixNum-ccl+1):
            # 定义结束位置 j
            j = i + ccl - 1
            cost[i][j] = NULLFLAG
            # 定义最佳中间分割点k的位置
            for k in range(i, j):
                tempCost = cost[i][k] + cost[k+1][j] + matrix[i][0] * matrix[k][1] * matrix[j][1]

                if tempCost < cost[i][j]:
                    cost[i][j] = tempCost
                    bestShot[i][j] = k
    return cost, bestShot

def printParens(bestShot, i, j):
    if i == j:
        print("A", i, end="")
    else:
        print("(", end='')
        k = bestShot[i][j]
        printParens(bestShot, i, k)
        printParens(bestShot, k+1, j)
        print(")", end='')


if __name__ == '__main__':
    Matrix = [
        [30, 35],
        [35, 15],
        [15, 5],
        [5, 10],
        [10, 20],
        [20, 25],
    ]
    cost, bestShot = MatrixChainOrder(Matrix)
    print(cost)
    print(bestShot)

    printParens(bestShot, 0, 5)


