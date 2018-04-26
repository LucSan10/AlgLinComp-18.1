import numpy as np
import GaussJordan as gj
import Subs as sub

def twoByTwoInv(matrix):
	
	if not (matrix.shape[0] == matrix.shape[1] == 2): raise ValueError("The matrix\n"+str(matrix)+"\nis not 2x2")
	
	invMatrix = np.copy(matrix)
	I = np.eye(2, dtype = bool)

	den = invMatrix[0,0]*invMatrix[1,1]-invMatrix[1,0]*invMatrix[0,1]
	invMatrix = invMatrix[[1,0]].T[[1,0]]
	invMatrix[I[[1,0]]] = -invMatrix[I[[1,0]]]
	invMatrix /= den

	return invMatrix

def leastSq(A):

	size = A.shape[0]
	X = A[:,0][:,np.newaxis]
	Y = A[:,1][:,np.newaxis]

	P = np.concatenate((np.ones((size,1)),X), axis = 1)
	A = P.T.dot(P)
	C = P.T.dot(Y)

	invA = twoByTwoInv(A)
	B = invA.dot(C)

	return B

def interPol(A):

	degree = A.shape[0]
	X = A[:,0][:,np.newaxis]
	Y = A[:,1][:,np.newaxis]

	P = np.concatenate([X**i for i in range(0,degree)], axis = 1)
	(D,C,M) = gj.gaussElim(P,Y)
	B = sub.backSub(D,C)

	return B