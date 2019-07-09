'''
已知前序遍历和中序遍历，重建二叉树
'''
class Node:
    def __init__(self, data, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right

class Tree:
    def __init__(self):
        self.root = None

    def rebuitTree(self, preOrder, inOrder):
        if set(preOrder) != set(inOrder):
            print('非对称子串')
            return
        if len(preOrder) == 0:
            return None
        if len(preOrder) == 1:
            node = Node(preOrder[0])
        else:
            root_index = preOrder.index(inOrder[0])
            node = Node(inOrder[0])
            node.left = self.rebuitTree(preOrder[0: root_index], inOrder[1: root_index+1])
            node.right = self.rebuitTree(preOrder[root_index+1:], inOrder[root_index+1:])
        return node

    def preOrder(self, node):
        if node == None:
            return
        print(node.data, end=', ')
        self.preOrder(node.left)
        self.preOrder(node.right)

    def midOrder(self, node):
        if node == None:
            return
        self.midOrder(node.left)
        print(node.data, end=', ')
        self.midOrder(node.right)

    def endOrder(self, node):
        if node == None:
            return
        self.endOrder(node.left)
        self.endOrder(node.right)
        print(node.data, end=', ')
if __name__ == '__main__':
    preOrder = [35, 37, 47, 50, 51, 62, 73, 88, 93, 99]
    inOrder =  [62, 47, 35, 37, 51, 50, 88, 73, 99, 93]

    myTree = Tree()
    myTree.root = myTree.rebuitTree(preOrder, inOrder)
    myTree.midOrder(myTree.root)
    print('')
    myTree.preOrder(myTree.root)
    print('')