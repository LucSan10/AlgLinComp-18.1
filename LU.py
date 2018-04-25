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


print("\n\n# Question 2\n")

A = np.array([[5,-4,1,0],[-4,6,-4,1],[1,-4,6,-4],[0,1,-4,5]], dtype = float)
B = np.array([[-1],[0],[1],[0]])

print("\nStarting matrices:\n\nA:\n")
print(A)
print("\nB:\n")
print(B)

(P,L,U,det) = LUDecomp(A)
print(det)

B = P.dot(B)
L = P.dot(L)

print("\nLet A = LU:")
print("\nL equals:\n" + str(L))
print("\nU equals:\n" + str(U))

print("\nProof:\n" + str(L.dot(U)))

print("Assuming that UX = Y, we have LY = B")
print("\nGiven that AX = B, then LUX = B")
Y = sub.frontSub(L,B)
X = sub.backSub(U,Y)

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

Res = np.empty((10,1))
for i in range(0,10,2):
	Res[i] = 12-abs(2*i-8)

(Pm,Lm,Um,det) = LUDecomp(Matrix)

Res = Pm.dot(Res)
Lm = Pm.dot(Lm)

print("\nLet A = LU; given that AX = B, then LUX = B")
print("Assuming that UX = Y, we have LY = B")
print("\nL equals:\n" + str(Lm))
print("\nU equals:\n" + str(Um))

print("\nProof:\n" + str(Lm.dot(Um)))

YFinal = sub.frontSub(Lm,Res)
XFinal = sub.backSub(Um,YFinal)

print("\nX equals:\n" + str(XFinal))
print("\nY equals:\n" + str(YFinal))
print("\nProof:\n" + str(Um.dot(XFinal)))
print("\nB equals:\n" + str(Res))
print("\nProof:\n" + str(Lm.dot(YFinal)))