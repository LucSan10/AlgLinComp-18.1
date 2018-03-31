import numpy as np

def gaussJordan(A, index):
	size = A.shape
	L = np.eye(size[0],size[1], dtype = float)
	U = np.copy(L)
	if (A[index,index]):
		L[index+1:, index] = -A[index+1:, index]/A[index,index]
		U = L.dot(A)
		return (L,U)
	return (-1,-1)