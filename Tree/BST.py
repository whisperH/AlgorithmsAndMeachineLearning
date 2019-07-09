'''
二叉排序树
'''

class BSTNode:
    def __init__(self, data, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right
    def __delete__(self):
        print('free BSTNode')

class BSTree:
    def __init__(self):
        self.root = None

    def isEmpty(self):
        if self.root is None:
            print('None Tree')
        return self.root is None

    def search(self, key, node, parent):
        if node is None:
            return False, node, parent
        elif key == node.data:
            return True, node, parent
        elif key > node.data:
            return self.search(key, node.right, node)
        elif key < node.data:
            return self.search(key, node.left, node)


    def insert(self, node, data):
        if node is None:
            node = BSTNode(data)
        elif node.data < data:
            node.right = self.insert(node.right, data)
        elif node.data > data:
            node.left = self.insert(node.left, data)
        else:
            print(node.data, ' is already in Tree')
        return node

    def delnode(self, data):
        flag, node, parent = self.search(data, self.root, self.root)
        if flag is False:
            print('not in Tree')
            return
        else:
            if (node.left is None) and (node.right is None):
                print('left and right tree is None')
                del node
            else:
                if node.left is None:
                    print('left is None')
                    parent.left = node.right
                    del node
                elif node.right is None:
                    print('right is None')
                    parent.left = node.left
                    del node
                else:
                    print('left and right is existing')
                    replace_node = node.left
                    pre = node
                    while replace_node.right is not None:
                        pre = replace_node
                        replace_node = replace_node.right
                    print('replace node is:', replace_node.data)
                    node.data = replace_node.data
                    if replace_node.left is not None:
                        pre.right = replace_node.left
                    else:
                        pre.left = replace_node.left

                    del replace_node

    def deleteNode(self, node):
        if (node.left is None) or (node.right is None):
            if node.left is None:
                tmp = node.right
                del node
                return tmp
            elif node.right is None:
                tmp = node.left
                del node
                return tmp
            else:
                del node
                return None
        else:
            replace_node = node.left
            if replace_node.right is None:
                node.left = replace_node.left
            else:
                pre = replace_node
                while replace_node.right is not None:
                    pre = replace_node
                    replace_node = replace_node.right
                pre.right = replace_node.left
            node.data = replace_node.data
            print("replace data is :", replace_node.data)
            del replace_node
            return node

    def selfDelNode(self, node, key):
        if node is None:
            print('not in Tree')
        else:
            print('current data is ', node.data)
            if node.data == key:
                print('start deleting...')
                node = self.deleteNode(node)
            elif node.data > key:
                print('go left')
                node.left = self.selfDelNode(node.left, key)
            else:
                print('go right')
                node.right = self.selfDelNode(node.right, key)
            return node
    def treeDepth(self, node):
        if node is None:
            return 0
        return max(self.treeDepth(node.left), self.treeDepth(node.right))+1

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


    def findMin(self, node):
        if not self.isEmpty():
            while node.left is not None:
                node = node.left
            return node.data
    def findMax(self, node):
        if not self.isEmpty():
            while node.right is not None:
                node = node.right
            return node.data
if __name__ == '__main__':
    datalist = [62, 58, 88, 47, 73, 99, 35, 51, 93, 37, 50]
    myTree = BSTree()
    for idata in datalist:
        myTree.root = myTree.insert(myTree.root, idata)
    print(myTree.treeDepth(myTree.root))

    print('before delete')
    myTree.midOrder(myTree.root)
    print('')
    myTree.preOrder(myTree.root)
    print('')
    # myTree.delnode(62)
    myTree.root = myTree.selfDelNode(myTree.root, 58)
    print('after delete')
    myTree.midOrder(myTree.root)
    print('')
    myTree.preOrder(myTree.root)
    print('')

    print("Min value is :", myTree.findMin(myTree.root))
    print("Max value is :", myTree.findMax(myTree.root))

