import numpy as np

arr = np.arange(12).reshape([6,2])
index = [True, True, False, False, False, False]
# replace = np.ones([1,6])
x = arr[index]
print(f"x = {x}")
print(f"arr[index] = {arr[index]}")
x[:,1] = 100
arr[index] = x
print(f"x = {x}")
print(f"arr[index] = {arr[index]}")


