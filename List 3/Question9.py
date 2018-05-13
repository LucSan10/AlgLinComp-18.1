import LeastSq as ls
import numpy as np

n = input("Please define the number of points:  ")
if int(n) < 2: raise ValueError("Can't fit straight line without at least 2 points")

arr = []

for i in range(int(n)):
    print("Please input an X value: ")
    x = input("X: ")
    print("Please input an Y value: ")
    y = input("Y: ")

    arr.append([float(x),float(y)])

A = np.array(arr)

B = ls.leastSq(A)

print("Starting matrices:\n\nX:\n")
print(A[:,0][:,np.newaxis])
print("\nY:\n")
print(A[:,1][:,np.newaxis])

print("\nB = [b0, b1]T so that XB = Y:\n")
print(B)