import numpy as np

def power(A):
	biggestEig = 1
	k = 0
	
	X = np.ones((A.shape[0],1))
	X = A.dot(X)
	while (X[k] == 1): k+=1
	while (np.all(abs((X[k]-biggestEig)/X[k]) > 0.00001)):
		biggestEig = np.copy(X[k])
		X /= biggestEig
		X = A.dot(X)
	return (biggestEig, X/biggestEig)