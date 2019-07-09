'''
堆排序：
1、构造堆，首先堆是一个完全二叉树，所以结点数目不存在奇数的情况.
    a.在调整最大堆时，仅调整所有子树的根节点（大->小调整）。
        若从0开始，即length//2-1，则孩子结点的下标分别为1，2(2*i+1, 2*i+2)
    b.注意传递的参数（数组，子树根节点标号，调整数组的长度）
    c.对每棵子树均寻找其子节点中最大的值，并与根节点值交换，注意节点标号的交换

2、调整大小
    a.最大堆中第一个元素最大值，与最数组最后的值进行交换
    b.动态维护一个数组长度为（array.length-j）的最大堆
    c.注意j的取值范围，调整数组中还剩一个元素时，排序完成。



'''

def HeapAdjust(data, sRoot, length):
    temp = data[sRoot]
    j = sRoot * 2 + 1
    while j < length:
        j = j if data[j] > data[j+1] else j+1
        if temp > data[j]:
            break
        data[sRoot] = data[j]
        sRoot = j
        j = j*2+1
    data[sRoot] = temp
    return data


def HeapSort(data):
    for i in range(len(data)//2-1, -1, -1):
        print(i)
        data = HeapAdjust(data, i, len(data))
    print('最大堆')
    print(data)
    print('#####################')
    for j in range(len(data)-1, -1, -1):
        data[0], data[j] = data[j], data[0]
        data = HeapAdjust(data, 0, j-1)
    return data



if __name__ == '__main__':
    # data = [
    #     12, 43, 4, 5, 32, 3, 14, 56, 6,
    #     2, 1, 63, 45, 6, 576, 3
    # ]
    data = [
        50, 10, 90, 30, 70, 40, 80, 60, 20, 100, 300
    ]
    print(HeapSort(data))
