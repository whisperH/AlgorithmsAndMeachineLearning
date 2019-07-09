'''
给定一个二叉树，计算整个树的坡度。
一个树的节点的坡度定义即为，该节点左子树的结点之和和右子树结点之和的差的绝对值。
空结点的的坡度是0。
整个树的坡度就是其所有节点的坡度之和。
https://leetcode-cn.com/problems/binary-tree-tilt/submissions/

注意节点的累积问题。
1、树的坡度定义为：整个树的坡度就是其所有节点的坡度之和。
那么对于父节点来说：左子树的坡度和+右子树的坡度和即为当前子树的坡度和

2、每个节点的坡度和：该节点左子树的结点之和和右子树结点之和的差的绝对值

'''
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class BiTree:
    def __init__(self):
        self.root = None

    def createBiTree(self, node, data, inum):
        if inum < len(data):
            if data[inum] == '#':
                node = None
            else:
                node = TreeNode(data[inum])
                node.left = self.createBiTree(node.left, data, inum * 2 + 1)
                node.right = self.createBiTree(node.right, data, inum * 2 + 2)
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
    def getTilt(self, root: TreeNode):
        if root is None:
            return 0, 0
        else:
            leftTilt, leftSum = Solution.getTilt(self, root.left)
            rightTilt, rightSum = Solution.getTilt(self, root.right)
            tilt = abs(
                leftTilt - rightTilt
            )
            # all = leftTilt + rightTilt
            print(root.val, 'tilt is :', tilt)
            print(root.val, 'leftTilt:', leftTilt)
            print(root.val, 'leftSum:', leftSum)
            print(root.val, 'rightSum:', rightSum)
            print(root.val, 'rightTilt:', rightTilt)
            return leftTilt + rightTilt + root.val, tilt + leftSum + rightSum

    def findTilt(self, root: TreeNode) -> int:
        nodeTilt, allTilt = self.getTilt(root)
        return allTilt


if __name__ == '__main__':
    data_list = [1, 2, 3, 4, "#", 5]
    # data_list = [1, 2, "#", 3, 4]
    btree = BiTree()

    btree.root = btree.createBiTree(btree.root, data_list, 0)
    btree.printTrave(btree.root)

    s = Solution()
    print(s.findTilt(btree.root))