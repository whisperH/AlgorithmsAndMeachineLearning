'''
归并排序:
-----------------------递归版本-----------------------
初始数组调用MergeSort函数，参数要求：原数组data、低位low、高位high
    a.递归终止条件“low==high”，返回一个仅包含一个元素的数组
    b.若不满足条件，则将数组按照中间数(low+high)/2分隔成为两个数组，
      递归调用MergeSort，传递的参数有所改变，data不变，
      改变的是low位和high位
    c.每轮递归调用结束，对LeftList 和 rightList返回的值进行排序，调用
      Msort函数（将两个有序的子序列，合并为一个序列），
      注意考虑leftList和rightList长度不相等的情况。


----------------------非递归版本-----------------------
其实就是申请一个新的数组，然后和原数组左右捣腾。
在一个数组中，每隔subLength个数字部分有序。
循环合并这些部分有序的数组，最后即为有序数组。


'''

def Merge(leftList, rightList):
    subList = []
    ileft = iright = 0
    while ileft < len(leftList) and iright < len(rightList):
        if leftList[ileft] < rightList[iright]:
            subList.append(leftList[ileft])
            ileft += 1
        elif leftList[ileft] > rightList[iright]:
            subList.append(rightList[iright])
            iright += 1
        else:
            subList.append(leftList[ileft])
            subList.append(rightList[iright])
            ileft += 1
            iright += 1
    if ileft < len(leftList):
        subList += leftList[ileft:]
    else:
        subList += rightList[iright:]
    return subList

# =======================递归版本===========================
def MergeSort(data):
    data = Msort(data, 0, len(data)-1)
    return data
def Msort(data, low, high):
    if low == high:
        return [data[low]]
    else:
        mid = (low+high) // 2
        leftList = Msort(data, low, mid)
        rightList = Msort(data, mid+1, high)
        mergeList = Merge(leftList, rightList)
        return mergeList

# =======================非递归版本===========================
def MergePass(origin, subListLength):
    i = 0
    target = []
    while i < len(origin):
        leftList = origin[i: i+subListLength]
        i = i+subListLength
        rightList = origin[i: i+subListLength]
        i = i+subListLength
        target += Merge(leftList, rightList)
    return target
def MergeSort2(data):
    k = 0
    while k < len(data):
        data = MergePass(data, k+1)
        k = 2*k+1
    return data


if __name__ == '__main__':
    data1 = [
        12, 43, 4, 5, 32, 3, 14, 56, 6,
        2, 1, 63, 45, 6, 576, 3
    ]

    data = [
        50, 10, 90, 30, 70, 40, 80, 60, 20, 11
    ]
    print(MergeSort2(data))