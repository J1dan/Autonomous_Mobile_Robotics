import numpy as np

arr = np.arange(12).reshape([6,2])
index = [True, True, False, False, False, False]
replace = np.ones([1,6])
arr = np.insert(arr,2,replace,axis=1)
print(arr)


