'''
最长回文子序列：回文是所有正序和逆序相同的非空字符串，例如：
1. 所有长度为1的字符歘
2. civic、racecar、character中的carac

现要求给定输入字符串的最长回文子序列
'''

def getPalidrome(string):
    matrix = []
    for i in range(len(string)+1):
        matrix.append([0] * (len(string)+1))

    for j in range(len(string)-1, -1, -1):
        for i in range(len(string)):
            if string[i] == string[j]:
                matrix[len(string)-j][i+1] = matrix[len(string)-j-1][i]+1

    print(matrix)
    return 0


if __name__ == '__main__':
    strings = [
        'a',
        'civic',
        'character'
    ]
    for istr in strings:
        print(getPalidrome(istr))