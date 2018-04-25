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
 
# Question 2)

#A = np.array([[1,1,1],[1,2,4],[1,3,9],[1,4,16]], dtype = float)
#A = np.array([[5,-4,1,0],[-4,6,-4,1],[1,-4,6,-4],[0,1,-4,5]], dtype = float)
A = np.array([[1,2],[3,4]])
#B = np.array([[1],[2],[9],[20]], dtype = float)
B = np.array([[-1],[0]], dtype = float)


print("Starting matrices:\n\nA:\n")
print(A)
print("\nB:\n")
print(B)

(D,B,M) = gaussElim(A,B,True)

Inv = inverse(A,B)

print("\nA^-1 is:\n" + str(Inv))
print("\nProof:\n" + str(Inv.dot(A)))
"""
X = sub.backSub(D,B)

print("\nM is:\n" + str(M))
print("\nLet DX = B")
print("\nD equals:\n" + str(D.astype(float)))
print("\nX equals:\n" + str(X))
print("\nB equals:\n" + str(B))
print("\nProof:\n" + str(D.dot(X)))
"""