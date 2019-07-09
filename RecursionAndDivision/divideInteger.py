'''
整数划分问题：
n：需要划分的正数
maxAddFactory：划分因子中数值最大的因子
a.=========================m无限制==============================
    1.当 n 或者 maxAddFactory 有一个小于1时，该问题无法划分，0种组合。
    2.当 n 或者 maxAddFactory 有一个为1时，就只有一种可能，即全由 1 组成
    3.当 n < maxAddFactory 时，此时最大的划分因子为 n 而非 maxAddFactory
    4.当 n = maxAddFactory 时，4.1 + 4.2 的结果
      4.1： n=n（这一种情况）
      4.2：divideInteger(n, n-1)
    5.当 n > maxAddFactory > 1 时, 5.1 + 5.2
      5.1： n 的分解因子中包括 maxAddFactory：
            divideInteger(n-maxAddFactory, maxAddFactory)
      5.2：n 的分解因子中不包括 maxAddFactory：
            divideInteger(n, maxAddFactory-1)

b.========================= 默认 maxAddFactory 为奇数==============================
    1.当 n 或者 maxAddFactory 有一个小于1时，该问题无法划分，0种组合。
    2.当 n 或者 maxAddFactory 有一个为1时，就只有一种可能，即全由 1 组成
    3.当 n < maxAddFactory 时，此时最大的划分因子为小于n的最大的奇数
    4.当 n = maxAddFactory 时（maxAddFactory和n为奇数），
      4.1 + 4.2 的结果
                  4.1： n=n（这一种情况）
                  4.2：divideInteger(n, n-2)
    5.当 n > maxAddFactory > 1 时, 5.1 + 5.2
      5.1： n 的分解因子中包括 maxAddFactory：
            divideInteger(n-maxAddFactory, maxAddFactory)
      5.2：n 的分解因子中不包括 maxAddFactory：
            divideInteger(n, maxAddFactory-2)

c.========================= maxAddFactory 不能重复==============================
    1.当 n 或者 maxAddFactory 有一个小于1时，该问题无法划分，0种组合。
    2.当 n == 1，return 1;
      当 n > maxAddFactory == 1 return 0
    3.当 n < maxAddFactory 时，此时最大的划分因子为 n 而非 maxAddFactory
    4.当 n = maxAddFactory 时，
      4.1 + 4.2 的结果
                  4.1： n=n（这一种情况）
                  4.2：divideInteger(n, n-1)
    5.当 n > maxAddFactory > 1 时, 5.1 + 5.2
      5.1： n 的分解因子中包括 maxAddFactory：
            divideInteger(n-maxAddFactory, maxAddFactory-1)
      5.2：n 的分解因子中不包括 maxAddFactory：
            divideInteger(n, maxAddFactory-1)
'''

def divideInteger(n, maxAddFactory):
    if (n < 1) or (maxAddFactory < 1):
        return 0
    elif (n == 1) or (maxAddFactory == 1):
        return 1
    elif n < maxAddFactory:
        return divideInteger(n, n)
    elif n == maxAddFactory:
        return divideInteger(n, n-1) + 1
    else:
        return divideInteger(n, maxAddFactory-1) + divideInteger(n-maxAddFactory, maxAddFactory)
if __name__ == '__main__':
    print(divideInteger(6, 6))