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

A = np.array([[3,2,0],[2,3,-1],[0,-1,3]], dtype = float)

print("\nStarting matrices:\n\nA:\n")
print(A)

(eig, X) = power(A)

print("\nA's highest eigenvalue is: "+ str(eig)+"\n")
print("\nIts corresponding eigenvector is:\n" +str(X))
print("\nAX is:\n"+str(A.dot(X)))
print("\nProof:\n"+str(eig*X))