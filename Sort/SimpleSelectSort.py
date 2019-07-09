'''
首先设置一个标志为，认定第一个数字为最小值，记录其索引，
然后与后续索引对应的值进行比较，在比较的过程中更新索引值。
每一轮完成后，检查索引值是否发生改变，若发生改变则交换数字

与冒泡排序的差距在于，虽然时间复杂度都为n^2，
但是减少了交换的次数，略优于冒泡排序
'''

def SimpleSelectSort(data):
    if len(data) == 0:
        return
    for i in range(len(data)):
        flag = i
        for j in range(i+1, len(data)):
            if data[flag] > data[j]:
                flag = j
        if flag != i:
            tmp = data[i]
            data[i] = data[flag]
            data[flag] = tmp
    return data


if __name__ == '__main__':
    data = [
        12, 43, 4, 5, 32, 3, 14, 56, 6,
        2, 1, 63, 3, 45, 6, 576, 3
    ]
    # print(SimpleSelectSort(data))
