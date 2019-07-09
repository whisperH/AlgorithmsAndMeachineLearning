class SteelPlan(object):
    def __init__(self, price, plan):
        self.price = price
        self.plan = plan
        self.cutNum = len(self.plan)

'''
切割钢条的问题：问题描述，一个长度为 n 的钢条，可以切割成任意段，
每一段的售价均有所不同，具体参考 priceList，问如何切割使得收益总和最大？

=========================  version 1：cutRod(priceList, length) =======================
可将问题理解： 切割长度为 i 的钢条与 length-i 钢条价格和。即
    sunPrice = priceList[i] + cutRod(priceList, length-i)
    其中，i 的取值从 [1, length] 均可。
    要取最大的sunPrice，则需要判断条件:当前长度的收益和切割后收益的大小比较
    maxPrice = max(maxPrice, sunPrice)
    返回 maxPrice

    递归结束条件：
    当length == 0 时，return 0.
'''
def cutRod(priceList: dict, rodLength: int) -> int:
    if rodLength <= 0:
        return 0
    else:
        price = 0
        for i in [_ for _ in priceList.keys() if _ <= rodLength]:
            price = max(
                price,
                priceList[i] + cutRod(priceList, rodLength-i)
            )
        return price


'''
# 仍然是  自顶向下的递归。 
考虑将计算过的内容存储下来，减少内存开销:
此时结束条件发生改变。
在计算完成子问题后，记得保存子问题的求解结果。
'''
def cutRodMemory(priceList: dict, rodLength: int, bestPrice: dict)-> list:
    if rodLength in bestPrice.keys():
        return bestPrice[rodLength]
    else:
        price = 0
        for i in [_ for _ in priceList.keys() if _ <= rodLength]:
            price = max(
                price,
                priceList[i] + cutRodMemory(priceList, rodLength-i, bestPrice)
            )
        bestPrice[rodLength] = price
        return price

'''
自下向上方法：
该方法考虑的不是长度为 n 的钢条怎么切分的问题。
其从最小长度钢条开始考虑，用bestPrice记录每一个长度的最高价格，
当需要计算新的长度时，从bestPrice中寻找最佳的组合，注意
还有，不一定分割成两条，可以是多条
分割长度(i，j的取值)。
相加后与当前不切分价格进行比较，
得出当前长度最佳切分方案，存入bestPrice
'''


# priceList：字典类型，散列表，用于存放每段钢条对应的价格
# bestPrice1：字典类型，用于存放当前最佳组合的价格
# length：钢条的长度
def cutRodAux(priceList, bestPrice1, length):
    bestPrice1[0] = 0
    for i in range(1, length+1):
        sumPrice = -1
        for j in [_ for _ in priceList.keys() if _ <= i]:
            sumPrice = max(sumPrice, priceList[j]+bestPrice1[i-j])
        bestPrice1[i] = sumPrice
    return bestPrice1[length]



'''

'''
def cutRodAuxPlan(priceList, bestPrice, length):
    bestPrice[0] = SteelPlan(0, [])
    for i in [_ for _ in priceList.keys() if _ < length]:
        combinePrice = SteelPlan(0, [])

        for j in [_ for _ in priceList.keys() if _ <= i]:
            t = bestPrice[i-j].price + priceList[j]
            s = bestPrice[i-j].plan + [j]
            combinePrice.price = combinePrice.price if t < combinePrice.price else t
            combinePrice.plan = combinePrice.plan if t < combinePrice.price else s

        bestPrice[i] = combinePrice
        bestPrice[i].cutNum = len(combinePrice.plan)
        # if combinePrice.price > priceList[i]:
        #     bestPrice[i].price = combinePrice.price
        #     bestPrice[i].plan = combinePrice.plan
        #     bestPrice[i].cutNum = len(combinePrice.plan)
        # else:
        #     bestPrice[i].price = priceList[i]
        #     bestPrice[i].plan = [i]
        #     bestPrice[i].cutNum = len([i])
    # print(i)
    return bestPrice[length]


if __name__ == '__main__':
    priceList = {
        1: 1,  2: 5,  3: 8,  4: 9,  5: 10,
        6: 17, 7: 17, 8: 20, 9: 24, 10: 30,
    }
    bestPrice = {}

    steelLengthRes = {
        1: 1, 2: 5, 3: 8, 4: 10,
        5: 13, 6: 17, 7: 18,
        8: 22, 9: 25, 10: 30
    }
    for length in steelLengthRes.keys():
        res = cutRod(priceList, length)
        print(res)
        # assert res == steelLengthRes[length], 'CutRod函数，长度'+str(length)+'判断出错'
        # res = cutRodMemory(priceList, length, bestPrice)
        # assert res == steelLengthRes[length], 'cutRodMemory函数，长度'+str(length)+'判断出错'

        res1 = cutRodAux(priceList, bestPrice, length)
        print(res1)
        # assert res1 == steelLengthRes[length], 'cutRodAux 函数，长度' + str(length) + '判断出错'
        # res = cutRodAuxPlan(priceList, bestPrice, length)
        # assert res.price == steelLengthRes[length], 'cutRodAux函数，长度'+str(length)+'判断出错'
        # print('length:,', length, '总价为', res.price, ',切割数量为：', res.cutNum)
        # print(res.plan)