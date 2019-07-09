def findMax(dataList):
    if len(dataList) == 1:
        return dataList[0]
    else:
        return max(dataList[0], findMax(dataList[1:]))

if __name__ == '__main__':
    datalist = [1, 2, 2, 3, 34, 4, 5]
    print(findMax(datalist))