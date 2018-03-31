import numpy as np
import GaussJordan as gj
import Pivot as p

def LUDecomp(A):
	size = A.shape
	P = np.eye(size[0], size[1], dtype = float)
	L = np.copy(P)
	U = np.copy(A)
	error = 0
	for i in range(size[0]-1):
		if not (U[i,i]):
			(tempP,error) = p.pivot(U, i, 1)
			U = tempP.dot(U)
			P = tempP.dot(P)
			L = P.dot(L.dot(tempP))
		if error:
			print("the matrix\n"+str(A.view())+"\ncannot be inverted")
			return
		(tempL, U) = gj.gaussJordan(U, i)
		L[i+1:, i] = -tempL.dot(L)[i+1:, i]
	return (P,L,U)

A = np.array([[1,2,4],[3,8,14],[2,6,13]], dtype = float)
B = np.array([[3,1],[4,2]], dtype = float)
C = np.array([[3,1],[-6,-4]], dtype = float)
D = np.array([[3,1,6],[-6,0,-16],[0,8,-17]], dtype = float)
E = np.array([[2,1,0,1],[2,1,2,3],[0,0,1,2],[-4,-1,0,2]])

(P,L,U) = LUDecomp(E)

print("A = PLU\nA is:\n" + str(E))
print("\nP equals:\n" + str(P))
print("\nL equals:\n" + str(L))
print("\nU equals:\n" + str(U))

print("\nProof:\n" + str(P.dot(L.dot(U))))
