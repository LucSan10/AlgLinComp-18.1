import numpy as np

def pivot(A, index1, rep):
	print(A.shape[0])
	if rep == A.shape[0]:
		error = 1
		return (A,error)
	index2 = index1+rep

	if A.shape[0] <= index2: index2 -= A.shape(0)
	
	if not (A[index2,index1]): (P,error) = pivot(A, index1, rep+1)
	else:
		P = np.eye(A.shape[0], A.shape[1])
		P[[index1,index2]] = P[[index2,index1]]
		return(P, error)