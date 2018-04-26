import JacobiEigen as jac
import PowerMethod as pm
import numpy as np
import Cholesky as ch
import Subs as sub

A = np.array([[3,2,0],[2,3,-1],[0,-1,3]], dtype = float)
B = np.array([[1],[-1],[1]], dtype = float)

#####################################################################################################################################

print("\n\n# Question 3 c\n")

A = np.array([[3,2,0],[2,3,-1],[0,-1,3]], dtype = float)
B = np.array([[1],[-1],[1]], dtype = float)

print("\nStarting matrices:\n\nA:\n")
print(A)

(eig, X) = pm.power(A)

print("\nA's highest eigenvalue is: "+ str(eig)+"\n")
print("\nIts corresponding eigenvector is:\n" +str(X))
print("\nAX is:\n"+str(A.dot(X)))
print("\nProof:\n"+str(eig*X))

#####################################################################################################################################

print("\n\n# Question 3 e\n")

print("\n\n# Jacobi:\n")

A = np.array([[3,2,0],[2,3,-1],[0,-1,3]], dtype = float)
B = np.array([[1],[-1],[1]], dtype = float)

print("\nStarting matrices:\n\nA:\n")
print(A)
print("\nB:\n")
print(B)
print("\n")

(eigVal, eigVec) = jac.jacobi(A)

print("\nA's eigenvalues, L, are:\n"+ str(eigVal[:, np.newaxis]))
print("\nIts corresponding eigenvectors, V, are:\n" +str(eigVec))
print("\nProof: AV = L*V\n")

for i in range(A.shape[0]):
	print("\nAX["+str(i+1)+"]:\n"+str((A.dot(eigVec[:,i]))[:,np.newaxis]))
	print("\n#"+str(i)+" eigen:\n" +str((eigVal[i]*eigVec[:,i])[:,np.newaxis]))

print("\nLet AX = B; VY = X, then AVY = B")
print("Multiplying by V.T on both sides, (V.T)AVY = (V.T)B, LY = (V.T)B")
print("Y = (L^-1)(V.T)B, X = VY = V(L^-1)(V.T)B")

X = jac.eigenSolve(eigVal, eigVec, B)

print("\nX is:\n"+str(X))
print("\nProof: AX aprox= B\n"+str(A.dot(X)))

#####################################################################################################################################

print("\n\n# Cholesky:\n")

A = np.array([[3,2,0],[2,3,-1],[0,-1,3]], dtype = float)
B = np.array([[1],[-1],[1]], dtype = float)

print("\nStarting matrices:\n\nA:\n")
print(A)
print("\nB:\n")
print(B)
print("\n")

(L,LT,det) = ch.choleskyDecomp(A)

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