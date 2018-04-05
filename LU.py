import numpy as np
import GaussJordan as gj
import Pivot as p
import BackSub as bs

# Question 1.a)

def LUDecomp(A):
	if np.linalg.cond(A) > 10**6: raise ValueError("the matrix\n\n"+str(A)+"\n\nis singular") 
	if A.shape[0] != A.shape[1]: raise ValueError("the matrix\n\n"+str(A)+"\n\ndoesn't have a LU decomposition")
	
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
	
		if error: raise ValueError("the matrix\n\n"+str(A)+"\n\ncannot be inverted")
		(tempL, U) = gj.gauss(U, i)
		L[i+1:, i] = -tempL.dot(L)[i+1:, i]
	
	return (P,L,U)

print("\n\n# Question 2\n")

A = np.array([[5,-4,1,0],[-4,6,-4,1],[1,-4,6,-4],[0,1,-4,5]], dtype = float)
B = np.array([[-1],[0],[1],[0]])

print("\nStarting matrices:\n\nA:\n")
print(A)
print("\nB:\n")
print(B)

(P,L,U) = LUDecomp(A)

B = P.dot(B)

print("\nLet A = PLU:")
print("\nP equals:\n" + str(P))
print("\nL equals:\n" + str(L))
print("\nU equals:\n" + str(U))

print("\nProof:\n" + str(P.dot(L.dot(U))))

print("Assuming that UX = Y, we have LY = B")
print("\nGiven that AX = B, then LUX = B")
Y = bs.backSub(L,B,upper=False)
X = bs.backSub(U,Y)

print("\nX equals:\n" + str(X))
print("\nY equals:\n" + str(Y))
print("\nProof:\n" + str(U.dot(X)))
print("\nB equals:\n" + str(B))
print("\nProof:\n" + str(L.dot(Y)))

print("\n\n# Question 3\n")

Matrix = np.empty((10,10))
for i in range(10):
	for j in range(10):
		if i != j: Matrix[i,j] = 10-np.abs(i-j)
		else: Matrix[i,i] = 19-np.abs(3-i)
Matrix = Matrix.astype(float)

Res = np.array([[4],[0],[8],[0],[12],[0],[8],[0],[4],[0]], dtype = float)

(Pm,Lm,Um) = LUDecomp(Matrix)
Res = Pm.dot(Res)

print("\nLet A = PLU; given that AX = B, then LUX = B")
print("Assuming that UX = Y, we have LY = B")
print("\nP equals:\n" + str(Pm))
print("\nL equals:\n" + str(Lm))
print("\nU equals:\n" + str(Um))

print("\nProof:\n" + str(Pm.dot(Lm.dot(Um))))

YFinal = bs.backSub(Lm,Res,upper=False)
XFinal = bs.backSub(Um,YFinal)

print("\nX equals:\n" + str(XFinal))
print("\nY equals:\n" + str(YFinal))
print("\nProof:\n" + str(Um.dot(XFinal)))
print("\nB equals:\n" + str(Res))
print("\nProof:\n" + str(Lm.dot(YFinal)))