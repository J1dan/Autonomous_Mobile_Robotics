import numpy as np

arr = np.arange(12).reshape([6,2])
arr1 = np.arange(10).reshape([10,1])
arr1[0:4] = -1
print(arr1[arr1>3])
arr2 = np.array([])

index = [True, True, False, False, False, False]
replace = np.ones([1,6])

# arr = np.insert(arr,2,replace,axis=1)

