'''
假设你要去露营，有一个容量为6磅的背包，有下列东西需要携带，怎样组合价值最大？
1. 水：3磅，10 value
2. 书：1磅，3 value
3. 食物：2磅，9 value
4. 夹克：2磅，5 vlaue
5. 相机：1磅，6 vlaue
'''

def bagging(bagWeight, thingList):
    bestSum = [0] * (bagWeight+1)
    for i in thingList.keys():
        for j in range(thingList[i]['weight'], bagWeight+1):
            bestSum[j] = max(
                bestSum[j],
                thingList[i]['value'] + bestSum[j-thingList[i]['weight']]
            )
    return bestSum



if __name__ == '__main__':
    thingList = {
        'water': {
            'weight': 3,
            'value': 10
        },
        'book': {
            'weight': 1,
            'value': 3
        },
        'food': {
            'weight': 2,
            'value': 9
        },
        'jacket': {
            'weight': 2,
            'value': 5
        },
        'camera': {
            'weight': 1,
            'value': 6
        }
    }
    bagList = []
    sumWeight = 6
    print(bagging(sumWeight, thingList))
