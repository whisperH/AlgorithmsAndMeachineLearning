def getLongCommonSeq(string1, string2):
    commonList = [0] * (len(string1) if len(string1) > len(string2) else len(string2))

    for istr in range(len(string1)):
        if string1[istr] == string2[0]:
            commonList[0] += 1
        for jstr in range(1, len(string2)):
            commonList[jstr] = max(
                commonList[jstr],
                commonList[jstr-1]
            )
            if string1[istr] == string2[jstr]:
                commonList[jstr] += 1
        print(commonList)
    return commonList

if __name__ == '__main__':
    string2 = 'clues'
    string1 = 'blue'
    getLongCommonSeq(string1, string2)