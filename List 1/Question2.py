import numpy as np
import Subs as sub
import Cholesky as ch
import GaussJordan as gj
import LU as lu

print("\n\n# Question 2 a\n")

A = np.array([[5,-4,1,0],[-4,6,-4,1],[1,-4,6,-4],[0,1,-4,5]], dtype = float)
B = np.array([[-1],[0],[1],[0]], dtype = float)

test = np.array([[2,1],[1,2]], dtype = float)

#####################################################################################################################################

print("\n\n# Gauss Decomposition:\n")

A = np.array([[5,-4,1,0],[-4,6,-4,1],[1,-4,6,-4],[0,1,-4,5]], dtype = float)
B = np.array([[-1],[0],[1],[0]], dtype = float)


print("Starting matrices:\n\nA:\n")
print(A)
print("\nB:\n")
print(B)
print("\n")

(D,B,M) = gj.gaussElim(A,B)

Inv = gj.inverse(A,B)

print("\nA^-1 is:\n" + str(Inv))
print("\nProof:\n" + str(Inv.dot(A)))

X = sub.backSub(D,B)

print("\nM is:\n" + str(M))
print("\nLet DX = B")
print("\nD equals:\n" + str(D.astype(float)))
print("\nX equals:\n" + str(X))
print("\nB equals:\n" + str(B))
print("\nProof:\n" + str(D.dot(X)))

#####################################################################################################################################

print("\n\n# Gauss-Jordan Decomposition:\n")

A = np.array([[5,-4,1,0],[-4,6,-4,1],[1,-4,6,-4],[0,1,-4,5]], dtype = float)
B = np.array([[-1],[0],[1],[0]], dtype = float)


print("Starting matrices:\n\nA:\n")
print(A)
print("\nB:\n")
print(B)
print("\n")

(D,B,M) = gj.gaussElim(A,B, True)

print("\nA^-1 is:\n" + str(Inv))
print("\nProof:\n" + str(Inv.dot(A)))

X = sub.backSub(D,B)

print("\nM is:\n" + str(M))
print("\nLet DX = B")
print("\nD equals:\n" + str(D.astype(float)))
print("\nX equals:\n" + str(X))
print("\nB equals:\n" + str(B))
print("\nProof:\n" + str(D.dot(X)))

#####################################################################################################################################

print("\n\n# LU Decomposition:\n")

A = np.array([[5,-4,1,0],[-4,6,-4,1],[1,-4,6,-4],[0,1,-4,5]], dtype = float)
B = np.array([[-1],[0],[1],[0]])

print("\nStarting matrices:\n\nA:\n")
print(A)
print("\nB:\n")
print(B)
print("\n")

(P,L,U,det) = lu.LUDecomp(A)
print(det)

B = P.dot(B)
L = P.dot(L)

print("\nLet A = LU:")
print("\nL equals:\n" + str(L))
print("\nU equals:\n" + str(U))

print("\nProof:\n" + str(L.dot(U)))

print("Assuming that UX = Y, we have LY = B")
print("\nGiven that AX = B, then LUX = B")
Y = sub.frontSub(L,B)
X = sub.backSub(U,Y)

print("\nX equals:\n" + str(X))
print("\nY equals:\n" + str(Y))
print("\nProof:\n" + str(U.dot(X)))
print("\nB equals:\n" + str(B))
print("\nProof:\n" + str(L.dot(Y)))

#####################################################################################################################################

print("# Cholesky Decomposition:\n")

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

#####################################################################################################################################

print("\n\n# Question 2 b\n")

print("Starting matrices:\n\nA:\n")
print(A)
print("\nB:\n")
print(B)
print("\n")

(D,B,M) = gj.gaussElim(A,B,True)

Inv = gj.inverse(A,B)

print("\nA^-1 is:\n" + str(Inv))
print("\nProof:\n" + str(Inv.dot(A)))

#####################################################################################################################################

print("\n\n# Question 2 c\n")

print("Starting matrices:\n\nA:\n")
print(A)
print("\nB:\n")
print(B)
print("\n")

(L,LT,det) = ch.choleskyDecomp(A)

print("Det(A) = " + str(det))
