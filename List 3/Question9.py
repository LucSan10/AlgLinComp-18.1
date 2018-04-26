import LeastSq as ls
import numpy as np

n = input("Please define the number of points:  ")

arr = []

for i in range(int(n)):
    print("Please input an X value: ")
    x = input("X: ")
    print("Please input an Y value: ")
    y = input("Y: ")

    arr.append([float(x),float(y)])

A = np.array(arr)

B = ls.leastSq(A)

print(A)
print("\n")
print(B)