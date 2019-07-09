'''
直接插入排序：
在遍历数组的过程中，若存在第i个数字乱序，则
将该数字提出来记为tmp：tmp = data[i]
前一个数字记为 j=i-1
在循环条件:
    (data[j]<tmp) and j>=0：
进行循环：前面的数字依次往后移动
        data[j+1] = data[j]
        j--
'''

def StraightInserSort(data):
    for i in range(1, len(data)):
        if data[i] < data[i-1]:
            tmp = data[i]
            j = i-1
            while (data[j] < tmp) and (j >= 0):
                data[j+1] = data[j]
                j = j-1
            data[j] = tmp
    return data

if __name__ == '__main__':
    data = [
        12, 43, 4, 5, 32, 3, 14, 56, 6,
        2, 1, 63, 3, 45, 6, 576, 3
    ]
    print(StraightInserSort(data))