import numpy as np

x = np.random.randint(100, size=(3,3))
print(x)

x_min = np.min(x)
arr = np.where(x == x_min)
zip(arr[0], arr[1])
print(arr)

x_max = np.max(x)
arr = np.where(x == x_max)
zip(arr[0], arr[1])
print(arr)

