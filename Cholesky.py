import numpy as np
import Subs as sub

# Question 1.b)

def choleskyDecomp(matrix):

	n = matrix.shape[0]
	(eigVal, eigVec) = np.linalg.eig(matrix)
	
	if not (np.all(matrix == matrix.T) and np.all(eigVal > 0)):
		raise ValueError("the matrix\n\n"+str(matrix)+"\n\ndoesn't have a Cholesky decomposition")
	L = np.zeros((n, n), dtype = float)

	for i in range(1, n+1):
		L[i-1,i-1] = (matrix[i-1,i-1] - choleskySum(i, i, L))**0.5

		for j in range (i+1, n+1):
			L[j-1,i-1] = (1 / L[i-1,i-1]) * (matrix[j-1,i-1] - choleskySum(j, i, L))

	det = (np.prod(np.diag(L)))**2
	return (L,L.T,det)

def choleskySum(lineIndex, columnIndex, L):
	temp = 0
	if (columnIndex - 1) == 0:
		return temp
	for i in range(0, columnIndex - 1):
		temp += L[lineIndex-1,i] * L[columnIndex-1,i]
	return temp


print("\n\n# Question 2\n")

A = np.array([[5,-4,1,0],[-4,6,-4,1],[1,-4,6,-4],[0,1,-4,5]], dtype = float)
B = np.array([[-1],[0],[1],[0]], dtype = float)

test = np.array([[2,1],[1,2]], dtype = float)

print("\nStarting matrices:\n\nA:\n")
print(A)
print("\nB:\n")
print(B)

(L,LT,det) = choleskyDecomp(A)
print(det)

print("\nLet A = LLt:")
print("\nL equals:\n" + str(L))
print("\nLt equals:\n" + str(LT))
print("\nL * Lt equals:\n" + str(L.dot(LT)))

print("\nGiven that AX = B, then LLtX = B")
print("Assuming that LtX = Y, we have LY = B")

Y = sub.frontSub(L,B)
X = sub.backSub(LT,Y)

print("\nX equals:\n" + str(X))
print("\nY equals:\n" + str(Y))
print("\nProof:\n" + str(LT.dot(X)))
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

(Lm,LTm) = choleskyDecomp(Matrix)

print("\nLet A = LLt:")
print("\nL equals:\n" + str(Lm))
print("\nLt equals:\n" + str(LTm))
print("\nL * Lt equals:\n" + str(Lm.dot(LTm)))

print("\nGiven that AX = B, then LLtX = B")
print("Assuming that LtX = Y, we have LY = B")

YFinal = sub.frontSub(Lm,Res)
XFinal = sub.backSub(LTm,YFinal)

print("\nX equals:\n" + str(XFinal))
print("\nY equals:\n" + str(YFinal))
print("\nProof:\n" + str(LTm.dot(XFinal)))
print("\nB equals:\n" + str(Res))
print("\nProof:\n" + str(Lm.dot(YFinal)))