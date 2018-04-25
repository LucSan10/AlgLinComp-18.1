import numpy as np

def jacobi(A):

	if not (np.all(A == A.T)):
		raise ValueError("the matrix\n\n"+str(A)+"\n\ncan't be processed by the\nJacobi Eigenvalue algorithm")
	
	n = A.shape[0]
	ignoreDiag = np.ones((n,n), dtype = bool)
	np.fill_diagonal(ignoreDiag,0)

	X = np.eye(n)
	R = np.copy(A)

	while not np.all(abs(R[ignoreDiag]) < 0.000001):
		
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
