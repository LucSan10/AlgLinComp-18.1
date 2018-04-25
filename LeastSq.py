import numpy as np

def twoByTwoInv(matrix):
	
	if not (A.shape[0] == A.shape[1] == 2): raise ValueError("The matrix\n"+str(matrix)+"\nis not 2x2")
	
	invMatrix = np.copy(matrix)
	I = np.eye(2)

	den = invMatrix[0,0]*invMatrix[1,1]-invMatrix[1,0]*invMatrix[0,1]
	invMatrix = invMatrix[[1,0]].T[[1,0]]
	invMatrix[I[[1,0]]] = -invMatrix[I[[1,0]]]
	invMatrix /= den

	return invMatrix

