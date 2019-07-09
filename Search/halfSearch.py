def halfSearch(datalist, key):
    low = 0
    high = len(datalist)
    while low <= high:
        mid = (low + high) // 2
        if key == datalist[mid]:
            return True
        elif key > datalist[mid]:
            low = mid + 1
        else:
            high = mid - 1
    return False