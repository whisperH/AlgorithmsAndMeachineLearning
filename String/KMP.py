'''
getNext值：其实就是
'''

def getNext(nextVal, patternList):
    i = 0
    j = -1
    while i < len(patternList)-1:
        if (j == -1) or (patternList[i] == patternList[j]):
            j += 1
            i += 1
            nextVal[i] = j

        else:
            j = nextVal[j]
    return nextVal
def KMP(oriList, patternList):
    nextVal = [-1] * len(patternList)
    nextVal = getNext(nextVal, patternList)
    print(nextVal)
    i = 0
    j = -1
    while i < len(oriList) and j < len(patternList):
        if j == -1 or oriList[i] == patternList[j]:
            i += 1
            j += 1
        else:
            j = nextVal[j]
    if j >= len(patternList):
        print(oriList[i-j: i])
        return True
    else:
        print('none')
        return False
if __name__ == '__main__':
    KMP('ababc12sdawd', 'abcd')