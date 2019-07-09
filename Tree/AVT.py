LH = 1
EH = 0
RH = -1

class AVTNode:
    def __init__(self, data, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right
        self.BF = 0

class AVTree:
    def __init__(self):
        self.root = None

    def getDepth(self, node):
        if node is None:
            return 0
        return max(self.getDepth(node.left), self.getDepth(node.right)) + 1

    def getBF(self, node):
        if node is None:
            return 0
        return self.getDepth(node.left)-self.getDepth(node.right)

    def leftBalance(self, node):
        print('node已经处于不平衡状态，需要进行左平衡操作：')
        child = node.left
        if self.getBF(child) == LH:
            print('node无右孩子，只进行右旋')
            node = self.rRotate(node)
        elif self.getBF(child) == RH:
            print('node有右孩子，进行双旋：')
            node.left = self.lRotate(child)
            node = self.rRotate(node)
        return node

    def rightBalance(self, node):
        print('node已经处于不平衡状态，需要进行右平衡操作：')
        child = node.right
        if self.getBF(child) == RH:
            print('node无左孩子，只进行右旋')
            node = self.lRotate(node)
        elif self.getBF(child) == LH:
            print('node有左孩子，进行双旋')
            node.right = self.rRotate(child)
            node = self.lRotate(node)
        return node

    def search(self, node, parent, key):
        if node is None:
            return False, node, parent
        elif node.data == key:
            return True, node, parent
        elif node.data < key:
            return self.search(node.right, node, key)
        else:
            return self.search(node.left, node, key)

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
            print('左右子树均不为NULL')
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


    def delNode(self, node, data):
        if node is None:
            print('Null Tree')
            return
        else:
            if node.data == data:
                print('deleting data', node.data)
                node = self.deleteNode(node)
            elif node.data > data:
                node.left = self.delNode(node.left, data)
            else:
                node.right = self.delNode(node.right, data)

            if node is not None:
                if self.getBF(node) == 2:
                    node = self.leftBalance(node)
                elif self.getBF(node) == -2:
                    node = self.rightBalance(node)
            return node

    def insert(self, node, data):
        if node is None:
            # print('create')
            node = AVTNode(data)
            # print('create back')
        elif node.data < data:
            # print('go right')
            # print(node.data)
            node.right = self.insert(node.right, data)
            # print('right back')
        elif node.data > data:
            # print('go left')
            # print(node.data)
            node.left = self.insert(node.left, data)
            # print('left back')
        else:
            print('data has been in Tree')

        print(node.data, "'s balance factory is: ", self.getBF(node))
        if self.getBF(node) == 2:
            node = self.leftBalance(node)
        elif self.getBF(node) == -2:
            node = self.rightBalance(node)
        return node

    def rRotate(self, node):
        next = node.left
        node.left = next.right
        next.right = node
        return next

    def lRotate(self, node):
        next = node.right
        node.right = next.left
        next.left = node
        return next

    def preOrder(self, node):
        if node == None:
            return
        print(node.data, end=' ')
        self.preOrder(node.left)
        self.preOrder(node.right)

    def midOrder(self, node):
        if node == None:
            return
        self.midOrder(node.left)
        print(node.data, end=' ')
        self.midOrder(node.right)

    def findMin(self, root):
        if root is None:
            print('Null Tree')
            return
        while root.left is not None:
            root = root.left
        return root

    def findMax(self, root):
        if root is None:
            return
        while root.right is not None:
            root = root.right
        return root
if __name__ == '__main__':
    data_list = [62, 58, 88, 47, 73, 99, 35, 51, 93, 37]
    myTree = AVTree()
    for idata in data_list:
        myTree.root = myTree.insert(myTree.root, idata)
    print('中序遍历')
    myTree.midOrder(myTree.root)
    print('')
    print('前序遍历')
    myTree.preOrder(myTree.root)
    print('')

    myTree.root = myTree.delNode(myTree.root, 62)
    print('中序遍历')
    myTree.midOrder(myTree.root)
    print('')
    print('前序遍历')
    myTree.preOrder(myTree.root)
    print('')
