'''
排列问题：
在求所有的排列问题时，解法如下：
1.perm函数中的参数
  k：序列开始下标（需要排列数组的开始位置）
  m：序列结束下标（需要排列数组的结束位置）
2.递归版本的思路：
  a.在数组初始情况下，依次固定住数组中的每一个数字，前缀+后缀排列即为结果
    a + (b,c,d)
    b + (a,c,d)
    c + (b,a,d)
    d + (b,c,a)
  因此需要遍历待排列数组中的所有元素，每轮遍历中需要调整元素的位置，
  固有swap(list, k, i)一说，i在遍历过程中依次指向 a,b,c,d

  b. 递归调用perm对子序列求解，此时子序列起始位置变为 k+1

  c. 递归完成后，再次循环时，需要将数组顺序调整至初始顺序。

  d. 递归结束条件：当初始位置和结束位置相同时，输出此时list中的各元素即可

'''

def perm(datalist, k, m):
    if k==m:
        for i in range(k):
            pass
            # print(datalist[i], end=',')
        print('')
        print('================')
    else:
      for i in range(k, m+1):
          datalist[k], datalist[i] = datalist[i], datalist[k]
          perm(datalist, k+1, m)
          print(datalist)
          print('change:', datalist[i], 'and', datalist[k])
          datalist[k], datalist[i] = datalist[i], datalist[k]



if __name__ == '__main__':
    perm([1, 2, 3, 4], 0, 3)