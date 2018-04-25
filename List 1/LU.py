import numpy as np
import GaussJordan as gj
import Pivot as p
import Subs as sub

# Question 1.a)

def LUDecomp(A):
	if np.linalg.cond(A) > 10**6: raise ValueError("the matrix\n\n"+str(A)+"\n\nis singular") 
	if A.shape[0] != A.shape[1]: raise ValueError("the matrix\n\n"+str(A)+"\n\ndoesn't have a LU decomposition")
	
	size = A.shape
	P = np.eye(size[0], size[1], dtype = float)
	L = np.copy(P)
	U = np.copy(A)
	det = 1
	error = 0

	for i in range(size[0]-1):
		if not (U[i,i]):
			(tempP,error) = p.pivot(U, i, 1)
			U = tempP.dot(U)
			P = tempP.dot(P)
			det *= -1
	
		if error: raise ValueError("the matrix\n\n"+str(A)+"\n\ncannot be inverted")
		(tempL, U) = gj.gauss(U, i)
		L[i+1:, i] = -tempL.dot(L)[i+1:, i]
	
	det *= np.prod(np.diag(U))
	return (P,L,U,det)