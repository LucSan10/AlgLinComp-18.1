import numpy as np

def choleskyDecomp(matrix):
	size = matrix.shape
	n = size[0]
	m = size[1]
	L = np.zeros((size[0], size[1]), dtype = float)

	for i in range(1, n+1):
		temp1 = matrix[i-1][i-1]
		temp2 = choleskySum(i, i, L)
		L[i-1][i-1] = np.sqrt(temp1 - temp2)
		#print("linha " + str(i) + " coluna " + str(i) + "-> " + str(temp1) + " - " + str(temp2) + " = " + str(L[i-1][i-1]) + "^2")
		#print(str(L) + "\n")
		
		for j in range (i+1, n+1):
			temp1 = matrix[j-1][i-1]
			temp2 = choleskySum(j, i, L)
			temp3 = L[i-1][i-1]
			temp4 = (1 / temp3) * (temp1 - temp2)
			L[j-1][i-1] = temp4
			#print("linha " + str(j) + " coluna " + str(i) + "-> " + "1 / " + str(temp3) + " * ("  + str(temp1) + " - " + str(temp2) + ") = " + str(L[j-1][i-1]))
			#print(str(L) + "\n")

	LT = np.transpose(L)

	return (L,LT)

def choleskySum(lineIndex, columnIndex, L):
	temp = 0
	if (columnIndex - 1) == 0:
		return temp
	for i in range(0, columnIndex - 1):
		temp += L[lineIndex-1][i] * L[columnIndex-1][i]
		#print(L[lineIndex-1][i])
		#print(L[columnIndex-1][i])

	#print (str(temp) + " soma")
	return temp

A = np.array([[1,0.2,0.4],[0.2,1,0.5],[0.4,0.5,1]], dtype = float)
B = np.array([[25,15,-5],[15,18,0],[-5,0,11]], dtype = float)
C = np.array([[18,22,54,42],[22,70,86,62],[54,86,174,134],[42,62,134,106]], dtype = float)



(L,LT) = choleskyDecomp(C)

print("A = L * LT\nA defined as:\n" +str(C))
print("\nL equals:\n" + str(L))
print("\nLT equals:\n" + str(LT))
print("\nL * LT equals:\n" + str(L.dot(LT)))