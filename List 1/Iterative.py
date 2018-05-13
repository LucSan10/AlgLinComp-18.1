import numpy as np

def residual(X0, X1):
	N1 = (X1-X0).T.dot((X1-X0))
	N0 = X1.T.dot(X1)
	return ((N1**0.5)/(N0**0.5))[0,0]

def calculateVector(A,X,B,index):
	
	size = A.shape
	ignoreIndex = np.ones((size[0],), dtype = bool)
	ignoreIndex[index] = 0
	minus = A[:,ignoreIndex].dot(X[ignoreIndex,:])
	x = (B[index,0]-minus)/A[index,index]
	return x

def jacobiIter(A,B):

	tol = 0.001
	size = A.shape
	if size[0] != size[1]: raise ValueError("The matrix\n"+str(A)+"\nis not square.")
	
	X1 = np.ones((size[0],1))
	X0 = np.copy(X1)

	for i in range(size[0]): X1[i,0] = calculateVector(A,X0,B,i)

	while (residual(X0,X1) < tol):
		X0 = np.copy(X1)
		for i in range(size[0]): X1[i,0] = calculateVector(A,X0,B,i)

	return X1

A = np.array([[5,-4,1,0],[-4,6,-4,1],[1,-4,6,-4],[0,1,-4,5]], dtype = float)
B = np.array([[-1],[0],[1],[0]])

print(jacobiIter(A,B))