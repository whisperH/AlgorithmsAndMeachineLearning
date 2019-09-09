import numpy as np
import math
def __calSD__(Xdata, Ydata, YLabel):
    a, b, c, d = 0, 0 ,0 ,0
    for i in range(Xdata.shape[0]):
        for j in range(i+1, Xdata.shape[0]):
            if Ydata[i] == Ydata[j]:
                if Ydata[i] == YLabel[j]:
                    a += 1
                else:
                    b += 1
            else:
                if Ydata[i] == YLabel[j]:
                    c += 1
                else:
                    d += 1
    return a, b, c, d

def JCIndex(Xdata, Ydata, YLabel):
    a, b, c, d = __calSD__(Xdata, Ydata, YLabel)
    return a/(a+b+c)

def FMI(Xdata, Ydata, YLabel):
    a, b, c, d = __calSD__(Xdata, Ydata, YLabel)
    return math.sqrt(a/(a+b)*a/(a+c))

def RI(Xdata, Ydata, YLabel):
    a, b, c, d = __calSD__(Xdata, Ydata, YLabel)
    m = Xdata.shape[0]
    return 2*(a+d)/(m*(m-1))
