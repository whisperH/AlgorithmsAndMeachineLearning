'''
将数组中的每一个元素都与后面的所有数进行比较，
如果大于后续的数字则进行交换
'''

def Bubble(data):
    for i in range(len(data)):
        for j in range(i, len(data)):
            if data[i] > data[j]:
                tmp = data[j]
                data[j] = data[i]
                data[i] = tmp
    return data


if __name__ == '__main__':
    data = [
        12, 43, 4, 5, 32, 3, 14, 56, 6,
        2, 1, 63, 3, 45, 6, 576, 3
    ]
    print(Bubble(data))