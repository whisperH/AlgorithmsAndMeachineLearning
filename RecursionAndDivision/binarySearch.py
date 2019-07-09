def recursiveBinarySearch(datalist, key, low, high):
    if low >= high:
        return False
    else:
        mid = (low+high)//2
        if datalist[mid] == key:
            return True
        elif datalist[mid] > key:
            return recursiveBinarySearch(datalist, key, low, mid)
        else:
            return recursiveBinarySearch(datalist, key, mid+1, high)

if __name__ == '__main__':
    datalist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    key = 10.2
    print(recursiveBinarySearch(datalist, key, 0, len(datalist)))