import numpy as np

def backSub(A, B):
	end = A.shape[0]
	X = np.zeros((end,1))
	X[end-1,0] = B[end-1,0]/A[end-1,end-1]

	for i in range(end-2,-1,-1):
		X[i,0] = (B[i,0] - (A[i,i+1:].dot(X[i+1:,0])))/A[i,i]
	return X

def frontSub(A, B):
	end = A.shape[0]
	X = np.zeros((end,1))
	X[0,0] = B[0,0]/A[0,0]

	for i in range(1,end,1):
		X[i,0] = (B[i,0] - (A[i,:i].dot(X[:i,0])))/A[i,i]
	return X

def diagSub(A,B):
	end = A.shape[0]
	X = np.zeros((end,1))
	X = (np.squeeze(B)/np.diag(A))[:,np.newaxis]
	return X