'''
自顶向下：
sum(a[0:n]) = a[0] + sum(a[1:n])
sum(a[1:n]) = a[1] + sum(a[2:n])
....

'''
def sumValue(dataList, startP):
    if startP == len(dataList)-1:
        return dataList[startP]
    else:
        value = dataList[startP] + sumValue(dataList, startP+1)
        return value

def sumValue2(dataList: list):
    if len(dataList) == 0:
        return 0
    else:
        sum = dataList[0] + sumValue2(dataList[1:])
        return sum



if __name__ == '__main__':
    dataList = [1, 2, 4, 5, 7]
    print(sumValue(dataList, 0))
    print(sumValue2(dataList))
