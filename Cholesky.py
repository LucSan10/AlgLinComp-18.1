import numpy as np

def choleskyDecomp(matrix):
	size = matrix.shape
	n = size[0]
	m = size[1]
	L = np.zeros((size[0], size[1]), dtype = float)

	for i in range(1, n+1):
		L[i-1][i-1] = np.sqrt(matrix[i-1][i-1] - choleskySum(i, i, L))

		for j in range (i+1, n+1):
			L[j-1][i-1] = (1 / L[i-1][i-1]) * (matrix[j-1][i-1] -choleskySum(j, i, L))
	LT = np.transpose(L)

	return (L,LT)

def choleskySum(lineIndex, columnIndex, L):
	temp = 0
	if (columnIndex - 1) == 0:
		return temp
	for i in range(0, columnIndex - 1):
		temp += L[lineIndex-1][i] * L[columnIndex-1][i]
	return temp

A = np.array([[1,0.2,0.4],[0.2,1,0.5],[0.4,0.5,1]], dtype = float)
B = np.array([[25,15,-5],[15,18,0],[-5,0,11]], dtype = float)
C = np.array([[18,22,54,42],[22,70,86,62],[54,86,174,134],[42,62,134,106]], dtype = float)



(L,LT) = choleskyDecomp(A)

print("A = L * LT\nA defined as:\n" +str(A))
print("\nL equals:\n" + str(L))
print("\nLT equals:\n" + str(LT))
print("\nL * LT equals:\n" + str(L.dot(LT)))