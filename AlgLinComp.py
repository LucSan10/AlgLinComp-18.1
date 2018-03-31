import numpy as np
import GaussJordan as gj
import Pivot as piv
from scipy import linalg

def LUDecomp(A):
	size = A.shape
	print(size)
	P = np.eye(size[0], size[1], dtype = float)
	L = np.copy(P)
	U = np.copy(A)
	error = False

	for i in range(size[0]-1):
		if not (U[i,i]):
			(tempP,error) = piv.pivot(U, i, 1)
			U = tempP.dot(U)
			P = tempP.dot(P)
		if error:
			print("the matrix\n"+str(A.view())+"\ncannot be inverted")
			return
		(tempL, U) = gj.gaussJordan(U, i)
		L[i+1:, i] = -tempL.dot(L)[i+1:, i]
	return (P,L,U)

A = np.array([[1,2,4],[3,8,14],[2,6,13]], dtype = float)
B = np.array([[3,1],[4,2]], dtype = float)
(P,L,U) = LUDecomp(B)

print(str(P)+"\n\n"+str(L)+"\n\n"+str(U))