'''
快速排序：
设置flag数字，
把小于flag的数字放到左边，
大于flag的数字放到右边。
返回排序后flag位的索引。

在递归调用时，注意结束条件：if low<high
在寻找flag标志位的时候：
    a.注意左右子序列的排序顺序，先高位后地位？？？？？？
    (
    若标志位选择的是低位，则应从高位序列开始循环，
    这是因为在高低位交换的过程中，flag标志位会随着high位移动，
    从选择low开始循环，flag位置并未随着low位的变化而变化
    )

=====================优化partitionIndex 的值=====================
取partitionIndex值为一个数组中左、中、右三个数的中位数，
然后保证low位为中位数的值
'''

def quickSort(data):
    data = Qsort1(data, 0, len(data)-1)
    return data

def getPartition(data, low, high):
    flag = data[low]
    while low < high:
        while (low < high) and (flag > data[low]):
            low += 1
        data[high], data[low] = data[low], data[high]
        while (low < high) and (flag <= data[high]):
            high -= 1
        data[high], data[low] = data[low], data[high]

    return data, low

# 优化不必要的交换
def getPartitionNoChange(data, low, high):
    flag = data[low]
    while low < high:
        while (low < high) and (flag < data[high]):
            high -= 1
        data[low] = data[high]
        while (low < high) and (flag >= data[low]):
            low += 1
        data[high] = data[low]
    data[low] = flag
    return data, low

# 优化index值
def getPartitionMid(data, low, high):
    m = (high+low)//2
    if data[low] > data[high]:
        data[low], data[high] = data[high], data[low]
    if data[m] > data[high]:
        data[low], data[high] = data[high], data[low]
    if data[m] < data[high]:
        data[m], data[low] = data[low], data[m]

    flag = data[low]
    while low < high:
        while (low < high) and (data[high] > flag):
            high -= 1
        data[high], data[low] = data[low], data[high]
        while (low < high) and (data[low] <= flag):
            low += 1
        data[high], data[low] = data[low], data[high]
    return data, low


def Qsort(data, low, high):
    if low < high:
        data, partitionIndex = getPartition(data, low, high)
        data = Qsort(data, low, partitionIndex-1)
        data = Qsort(data, partitionIndex+1, high)
        print(data)
    return data

def Qsort1(data, low, high):
    while low < high:
        # print(low)
        # print(high)
        data, partitionIndex = getPartitionMid(data, low, high)
        data = Qsort1(data, low, partitionIndex-1)
        # print(data)
        low = partitionIndex + 1
    return data


def Qsort2(datalist):
    if len(datalist) <= 1:
        return datalist
    else:
        povit = datalist[0]
        leftList = [e for e in datalist[1:] if e <= povit]
        rightList = [e for e in datalist[1:] if e > povit]
        return Qsort2(leftList) + [povit] + Qsort2(rightList)

if __name__ == '__main__':
    data1 = [
        12, 43, 4, 5, 32, 3, 14, 56, 6,
        2, 1, 63, 45, 6, 576, 3
    ]

    mydata = [
        50, 10, 90, 30, 70, 40, 80, 60, 20, 100, 300
    ]
    searchData = Qsort(data1, 0, len(data1)-1)
    print(searchData)
