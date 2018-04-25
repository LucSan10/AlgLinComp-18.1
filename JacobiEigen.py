import numpy as np

def jacobi(A):

	if not (np.all(A == A.T)):
		raise ValueError("the matrix\n\n"+str(A)+"\n\ncan't be processed by the\nJacobi Eigenvalue algorithm")
	
	n = A.shape[0]
	ignoreDiag = np.ones((n,n), dtype = bool)
	np.fill_diagonal(ignoreDiag,0)

	X = np.eye(n)
	R = np.copy(A)

	while not np.all(abs(R[ignoreDiag]) < 0.00001):
		
		index = np.argmax(abs(R[ignoreDiag]))
		index += 1+1*(int(index/n))
		i = int(index/n)
		j = index%n
		
		if R[i,i] == R[j,j]: fi = np.pi/4.0
		else: fi = 0.5*(np.arctan((2*R[i,j])/(R[i,i]-R[j,j])))

		P = np.eye(n)

		P[i,i] = np.cos(fi)
		P[j,j] = np.cos(fi)
		P[i,j] = np.sin(fi)
		P[j,i] = -np.sin(fi)
		if j>i:
			P[i,j] *= -1
			P[j,i] *= -1

		R = P.T.dot(R.dot(P))
		X = X.dot(P)

	return (np.diag(R),X)

def eigenSolve(L, V, B):

	invL = np.diag(1/L)
	R = V.T.dot(B)
	Y = invL.dot(R)
	X = V.dot(Y)
	return X

#A = np.array([[1,0.2,0],[0.2,1,0.5],[0,0.5,1]], dtype = float)
A = np.array([[3,2,0],[2,3,-1],[0,-1,3]], dtype = float)

#B = np.array([[1.2],[1.7],[1.5]], dtype = float)
B = np.array([[1],[-1],[1]], dtype = float)

print("\nStarting matrices:\n\nA:\n")
print(A)
print("\nB:\n")
print(B)

(eigVal, eigVec) = jacobi(A)

print("\nA's eigenvalues, L, are:\n"+ str(eigVal[:, np.newaxis]))
print("\nIts corresponding eigenvectors, V, are:\n" +str(eigVec))
print("\nProof: AV = L*V\n")

for i in range(A.shape[0]):
	print("\nAX["+str(i+1)+"]:\n"+str((A.dot(eigVec[:,i]))[:,np.newaxis]))
	print("\n#"+str(i)+" eigen:\n" +str((eigVal[i]*eigVec[:,i])[:,np.newaxis]))

print("\nLet AX = B; VY = X, then AVY = B")
print("Multiplying by V.T on both sides, (V.T)AVY = (V.T)B, LY = (V.T)B")
print("Y = (L^-1)(V.T)B, X = VY = V(L^-1)(V.T)B")

X = eigenSolve(eigVal, eigVec, B)

print("\nX is:\n"+str(X))
print("\nProof: AX aprox= B\n"+str(A.dot(X)))
