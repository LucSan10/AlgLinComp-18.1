import numpy as np
import Pivot as p
import Subs as sub

def gauss(A, index, upper = False):
	size = A.shape
	M = np.eye(size[0],size[1], dtype = float)
	A1 = np.copy(M)
	if A[index,index]:
		if upper: M[:index, index] = -A[:index, index]/A[index, index]
		else: M[index+1:, index] = -A[index+1:, index]/A[index,index]
		A1 = M.dot(A)
		return (M,A1)
	return (-1,-1)

def gaussElim(A, B, jordan = False):
	error = 0
	size = A.shape
	P = np.eye(size[0], size[1], dtype = float)
	M = np.copy(P)
	D = np.copy(A)

	for i in range(size[0]-1):
		if not (D[i,i]):
			(tempP,error) = p.pivot(D, i, 1)
			D = tempP.dot(D)
			P = tempP.dot(P)
		if error: raise ValueError("the matrix\n\n"+str(A)+"\n\ncannot be inverted")
		(tempM, D) = gauss(D, i)
		B = tempM.dot(B)
		M = tempM.dot(M)

	if jordan:
		for i in range(1, size[0]):
			if not (D[i,i]):
				(tempP,error) = p.pivot(D, i, 1)
				D = tempP.dot(D)
				P = tempP.dot(P)
			if error: raise ValueError("the matrix\n\n"+str(A)+"\n\ncannot be inverted")
			(tempM, D) = gauss(D, i, True)
			B = tempM.dot(B)
			M = tempM.dot(M)

	return (D, B, M)

def inverse(A, B):
	(D,B,M) = gaussElim(A,B,True)
	diag = np.diag(1/np.diag(D))
	return diag.dot(M)