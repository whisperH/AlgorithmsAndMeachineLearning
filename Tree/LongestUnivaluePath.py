'''
给定一个二叉树，找到最长的路径，这个路径中的每个节点具有相同值。
这条路径可以经过也可以不经过根节点。

'''
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

'''
创建二查树时，与创建二叉排序树略有不同，二叉排序树可以根据数据的大小决定左右方向，
二叉树则不同，所以传递的参数必须要指定左节点与右节点的数值。
另外，注意递归结束的条件
'''
class BinTree(object):
    def __init__(self):
        self.root = None

    def createBiTree(self, node, data, i):
        if i < len(data):
            if data[i] == '#':
                return None
            else:
                node = TreeNode(data[i])
                node.left = self.createBiTree(node.left, data, 2*i+1)
                node.right = self.createBiTree(node.right, data, 2*i+2)
            return node

    #先序遍历函数
    def preOrderTrave(self, bt):
        if bt is not None:
            print(bt.val, end=" ")
            self.preOrderTrave(bt.left)
            self.preOrderTrave(bt.right)
    #中序遍历函数
    def inOrderTrave(self, bt):
        if bt is not None:
            self.inOrderTrave(bt.left)
            print(bt.val, end=" ")
            self.inOrderTrave(bt.right)

    #后序遍历函数
    def postOrderTrave(self, bt):
        if bt is not None:
            self.postOrderTrave(bt.left)
            self.postOrderTrave(bt.right)
            print(bt.val, end=" ")

    #综合打印
    def printTrave(self, bt):
        print("先序遍历: ", end="")
        self.preOrderTrave(bt)
        print('\n')
        print("中序遍历: ", end="")
        self.inOrderTrave(bt)
        print('\n')
        print("后序遍历: ", end="")
        self.postOrderTrave(bt)
        print('\n')


class Solution:
    '''
    要点：
    1、保存最大值
    2、当前节点一定是根节点
    '''
    def longestUnivaluePath(self, root: TreeNode) -> int:
        self.maxLen = 0
        def countNum(root):
            if root is None:
                return 0
            else:
                leftLen = countNum(root.left)
                rightLen = countNum(root.right)
                lNodeLen = 0
                rNodeLen = 0
                if root.left and root.left.val == root.val:
                    lNodeLen = 1 + leftLen
                if root.right and root.right.val == root.val:
                    rNodeLen = 1 + rightLen
                self.maxLen = max(self.maxLen, lNodeLen + rNodeLen)
                return max(lNodeLen, rNodeLen)
        countNum(root)
        return self.maxLen

    def binaryTreePath(self, root):
        if root is None:
            return []
        if (root.left is None) and (root.right is None):
            return [root.val]
        else:
            path = []
            if root.left:
                path += self.binaryTreePath(root.left)
            if root.right:
                path += self.binaryTreePath(root.right)
            for index, val in enumerate(path):
                path[index] = str(root.val) + '->' + str(val)
            return path

if __name__ == '__main__':
    data_list = [5, 4, 5, 1, 2, "#", 5]
    # data_list = [1, 2, "#", 3, 4]
    btree = BinTree()
    btree.root = btree.createBiTree(btree.root, data_list, 0)
    btree.printTrave(btree.root)

    s = Solution()
    path = []
    # print(s.binaryTreePath(btree.root))
    print(s.longestUnivaluePath(btree.root))