import numpy as np

def backSub(A, B, diag = False, upper = True):
	end = A.shape[0]
	X = np.zeros((end,1))
	if diag: X = (np.squeeze(B)/np.diag(A))[:,np.newaxis]

	else:
		if upper:
			inicio = end-1
			fim = -1
		else:
			inicio = 0
			fim = end
	
		X[inicio,0] = B[inicio,0]/A[inicio,inicio]
		step = 1-(2*upper)
	
		for i in range(inicio+step,fim,step):
			if upper: X[i,0] = (B[i,0] - (A[i,i+1:].dot(X[i+1:,0])))/A[i,i]
			else: X[i,0] = (B[i,0] - (A[i,:i].dot(X[:i,0])))/A[i,i]
	return X