import numpy as np
import Subs as sub

def choleskyDecomp(matrix):

	n = matrix.shape[0]
	(eigVal, eigVec) = np.linalg.eig(matrix)
	
	if not (np.all(matrix == matrix.T) and np.all(eigVal > 0)):
		raise ValueError("the matrix\n\n"+str(matrix)+"\n\ndoesn't have a Cholesky decomposition")
	L = np.zeros((n, n), dtype = float)

	for i in range(1, n+1):
		L[i-1,i-1] = (matrix[i-1,i-1] - choleskySum(i, i, L))**0.5

		for j in range (i+1, n+1):
			L[j-1,i-1] = (1 / L[i-1,i-1]) * (matrix[j-1,i-1] - choleskySum(j, i, L))

	det = (np.prod(np.diag(L)))**2
	return (L,L.T,det)

def choleskySum(lineIndex, columnIndex, L):
	temp = 0
	if (columnIndex - 1) == 0:
		return temp
	for i in range(0, columnIndex - 1):
		temp += L[lineIndex-1,i] * L[columnIndex-1,i]
	return temp