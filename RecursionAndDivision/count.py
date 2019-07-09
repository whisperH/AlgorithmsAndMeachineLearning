def countNum(datalist):
    if datalist == []:
        return 0
    else:
        return 1 + countNum(datalist[1:])

if __name__ == '__main__':
    datalist = [1, 2, 3, 2, 4, 5, 66, 5, 77, 4, 3, 2, 23]
    print(len(datalist))
    print(countNum(datalist))