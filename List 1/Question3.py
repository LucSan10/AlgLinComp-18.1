import LU as lu
import Subs as sub
import numpy as np

print("\n\n# Question 3\n")

Matrix = np.empty((10,10))
for i in range(10):
	for j in range(10):
		if i != j: Matrix[i,j] = 10-np.abs(i-j)
		else: Matrix[i,i] = 19-np.abs(3-i)
Matrix = Matrix.astype(float)

Res = np.array([[4],[0],[8],[0],[12],[0],[8],[0],[4],[0]])

(Pm,Lm,Um,det) = lu.LUDecomp(Matrix)

Res = Pm.dot(Res)
Lm = Pm.dot(Lm)

print("\nA equals:\n" + str(Matrix))
print("\nB equals:\n" + str(Res))

print("\nL equals:\n" + str(Lm))
print("\nU equals:\n" + str(Um))

print("\nProof (LU = A):\n" + str(Lm.dot(Um)))

print("\nLet A = LU; given that AX = B, then LUX = B")
print("Assuming that UX = Y, we have LY = B")
YFinal = sub.frontSub(Lm,Res)
XFinal = sub.backSub(Um,YFinal)

print("\nX equals:\n" + str(XFinal))
print("\nY equals:\n" + str(YFinal))
print("\nProof (UX = Y):\n" + str(Um.dot(XFinal)))
print("\nB equals:\n" + str(Res))
print("\nProof (LY = B):\n" + str(Lm.dot(YFinal)))