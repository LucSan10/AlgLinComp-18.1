import numpy as np
import Cholesky as ch
import JacobiEigen as jac
import GaussJordan as gj
import PowerMethod as pm
import LU as lu
import Subs as sub
import LeastSq as lsq

A = np.array([[7.875, 4.375, 2.625, 0.875, 1.75, 0.875],[4.375, 8.75, 4.375, 2.625, 0.875, 1.75],[2.625, 4.375, 7.875, 4.375, 0.875, 1.75],[0.875, 2.625, 4.375, 5.25, 0.875, 1.75], [1.75, 0.875, 0.875, 0.875, 4.375, 2.625], [0.875, 1.75, 1.75, 1.75, 2.625, 3.5]])
B = np.array([[30],[10],[10],[-10],[0],[5]])

print("\n\n### Cholesky Decomposition ###\n\n")

(L,LT,det) = ch.choleskyDecomp(A)

print("\nLet A = LLt:")
print("\nL equals:\n" + str(L))
print("\nLt equals:\n" + str(LT))
print("\nProof (LLt = A):\n" + str(L.dot(LT)))
print("\ndet(A) = " + str(det))

print("\nGiven that AX = B, then LLtX = B")
print("Assuming that LtX = Y, we have LY = B")

Y = sub.frontSub(L,B)
X = sub.backSub(LT,Y)

print("\nX equals:\n" + str(X))
print("\nY equals:\n" + str(Y))
print("\nProof (LtX = Y):\n" + str(LT.dot(X)))
print("\nB equals:\n" + str(B))
print("\nProof (LY = B):\n" + str(L.dot(Y)))

##############################################################################################################

print("\n\n### LU Decomposition ###\n\n")

A = np.array([[7.875, 4.375, 2.625, 0.875, 1.75, 0.875],[4.375, 8.75, 4.375, 2.625, 0.875, 1.75],[2.625, 4.375, 7.875, 4.375, 0.875, 1.75],[0.875, 2.625, 4.375, 5.25, 0.875, 1.75], [1.75, 0.875, 0.875, 0.875, 4.375, 2.625], [0.875, 1.75, 1.75, 1.75, 2.625, 3.5]])
B = np.array([[30],[10],[10],[-10],[0],[5]])

(P,L,U,det) = lu.LUDecomp(A)

B = P.dot(B)
L = P.dot(L)

print("\nLet A = LU:")
print("\nL equals:\n" + str(L))
print("\nU equals:\n" + str(U))

print("\nProof (A = LU):\n" + str(L.dot(U)))
print("\ndet(A) = " + str(det))

print("\nGiven that AX = B, then LUX = B")
print("Assuming that UX = Y, we have LY = B")
Y = sub.frontSub(L,B)
X = sub.backSub(U,Y)

print("\nX equals:\n" + str(X))
print("\nY equals:\n" + str(Y))
print("\nProof (UX = Y):\n" + str(U.dot(X)))
print("\nB equals:\n" + str(B))
print("\nProof (LY = B):\n" + str(L.dot(Y)))

##############################################################################################################

print("\n\n### Eigenvalues and Eigenvectors ###\n\n")

A = np.array([[7.875, 4.375, 2.625, 0.875, 1.75, 0.875],[4.375, 8.75, 4.375, 2.625, 0.875, 1.75],[2.625, 4.375, 7.875, 4.375, 0.875, 1.75],[0.875, 2.625, 4.375, 5.25, 0.875, 1.75], [1.75, 0.875, 0.875, 0.875, 4.375, 2.625], [0.875, 1.75, 1.75, 1.75, 2.625, 3.5]])
B = np.array([[30],[10],[10],[-10],[0],[5]])

print("\nStarting matrices:\n\nA:\n")
print(A)
print("\nB:\n")
print(B)
print("\n")

(eigVal, eigVec, iteration) = jac.jacobi(A)

print("\nA's eigenvalues, L, are:\n"+ str(eigVal[:, np.newaxis]))
print("\nIts corresponding eigenvectors, V, are:\n" +str(eigVec))
print("\ndet(A) = " + str(np.prod(eigVal)))
print("\nThe algorithm undertook " +str(iteration)+ " iterations")

print("\nProof (AVi aprox= Li*Vi)\n")

for i in range(A.shape[0]):
	print("\nAX["+str(i+1)+"]:\n"+str((A.dot(eigVec[:,i]))[:,np.newaxis]))
	print("\n#"+str(i+1)+" eigen:\n" +str((eigVal[i]*eigVec[:,i])[:,np.newaxis])+"\n")

print("\nLet AX = B; VY = X, then AVY = B")
print("Multiplying by V.T on both sides, (V.T)AVY = (V.T)B, LY = (V.T)B")
print("Y = (L^-1)(V.T)B, X = VY = V(L^-1)(V.T)B")

X = jac.eigenSolve(eigVal, eigVec, B)

print("\nX is:\n"+str(X))
print("\nProof (AX aprox= B):\n"+str(A.dot(X)))

##############################################################################################################

print("\n\n### Least-Squares ###\n\n")

A = np.array([[-2.7, 3.75],[-1, 5.75],[0,7.5],[1,9.375],[1.6,10.625],[3.1,11.875]])

B = lsq.leastSq(A)

print("Starting matrices:\n\nX:\n")
print(A[:,0][:,np.newaxis])
print("\nY:\n")
print(A[:,1][:,np.newaxis])

print("\nB = [b0, b1]T so that XB = Y:\n")
print(B)