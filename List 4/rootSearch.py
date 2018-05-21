import time
import numpy as np
import sympy as sp
import GaussJordan as gj
import Subs as sub

def derivative(function, a, i = None, delta = 0.0001):
	if type(a) == np.ndarray:
		if i == None: i = a.size-1
		if a.dtype != float: a = a.astype(float)

		vx, vy = np.copy(a), np.copy(a)
		vx[i], vy[i] = vx[i]+delta, vy[i]-delta
		B = (function(vx)-function(vy))/(2*delta)
		
		if i == 0: return B
		recB = derivative(function, a, i-1)
		recB = np.concatenate((recB, B), axis = 1)
		
		return recB

	else:	
		front = function(a+delta)
		back = function(a-delta)
		return (front-back)/(2*delta)



def richardExtrapol(function, a, b = 0.5, q = 2):
	d1 = derivative(function, a, delta = 0.5)
	d2 = derivative(function, a, delta = 0.5/q)
	return d1 + (d1-d2)/((1/q)-1)



def bisecRoot(function, start, end, tol = 0.001):
	if function(start)*function(end) > 0:
		raise ValueError("Interval limits a and b have same signs.")
	
	root = abs(end+start)/2.0
	value = function(root)
	i = 1

	while value != 0 and abs(end-start)/2.0 > tol:
		if value > 0: end = root
		else: start = root
		root = abs(end+start)/2.0
		value = function(root)
		i += 1
	return (root,i)



def newtonRoot(function, x0, tol = 0.0001, reps = 1000):
	for i in range(reps):
		x1 = x0 - function(x0)/derivative(function,x0)
		
		if abs(x1-x0) < tol:
			return (x1,i)
		x0 = x1
	
	raise ValueError("newtonRoot() method with given x0 = "+str(x0)+" did not reach convergence.")



def secantRoot(function, x0, tol = 0.0001, reps = 1000, delta = 0.01):
	x1 = x0+delta
	lastVal = function(x0)
	x2 = 0
	
	for i in range(reps):
		currentVal = function(x1)
		x2 = x1 - (currentVal*(x1-x0))/(currentVal-lastVal)
		if abs(x2-x1) < tol: return (x2,i)
		
		lastVal = currentVal
		x0 = x1
		x1 = x2
	
	raise ValueError("secantRoot() method with given x0 = "+str(x0)+" and delta = "+str(delta)+" did not reach convergence.")



def inverseInterpolRoot(function, x1, x2, x3, tol = 0.0001, reps = 1000):
	l = [x1,x2,x3]
	l.sort()
	[x1,x2,x3] = l
	x0 = np.finfo(float).max
	
	for i in range(reps):
		y1 = function(x1)
		y2 = function(x2)
		y3 = function(x3)
		
		fi1 = 1.0*(y2*y3*x1)/((y1-y2)*(y1-y3))
		fi2 = 1.0*(y1*y3*x2)/((y2-y1)*(y2-y3))
		fi3 = 1.0*(y1*y2*x3)/((y3-y1)*(y3-y2))
		
		x4 = fi1 + fi2 + fi3
		if abs(x4-x0) < tol: return (x4,i)
		
		l[np.argmax(np.abs(l))] = x4
		l.sort()
		[x1,x2,x3] = l
		x0 = x4-x03

	raise ValueError("inverseInterpolRoot() method with starting x1, x2, x3 = ["+str(x1)+", "+str(x2)+", "+str(x3)+"] did not reach convergence")



def newtonMultiEq(functions, tol = 0.0001, reps = 1000):
	size = len(functions)
	symbolFunctions = sp.Matrix(sp.sympify(functions))
	symbolList = sp.symbols("x1:%d"%(size+1))
	
	fun = sp.lambdify([symbolList], symbolFunctions)
	X = np.ones((size,1))
	
	for i in range(reps):

		F = fun(X)
		J = derivative(fun, X)

		(U,B,L) = gj.gaussElim(J,-F)
		DeltaX = sub.backSub(U,B)
		
		X += DeltaX
		N1 = X.T.dot(X)**0.5
		NDelta = DeltaX.T.dot(DeltaX)**0.5
		
		if NDelta/N1 < tol: return (X,i)

	raise ValueError("newtonMultiEq() method with starting vector X:\n"+str(np.ones((size,1)))+"\n\ndid not reach convergence.")



def broydenMultiEq(functions, tol = 0.0001, reps = 1000):
	size = len(functions)
	symbolFunctions = sp.Matrix(sp.sympify(functions))
	symbolList = sp.symbols("x1:%d"%(size+1))
	
	fun = sp.lambdify([symbolList], symbolFunctions)
	X = np.array([[0.5*i] for i in range(size)])
	J = np.array([[0.5*i+0.5*j for i in range(size)] for j in range(size-1, -1, -1)])
	F0 = fun(X)

	for i in range(reps):

		(U,B,L) = gj.gaussElim(J,-F0)
		DeltaX = sub.backSub(U,B)

		X += DeltaX
		N1 = X.T.dot(X)**0.5
		NDelta = DeltaX.T.dot(DeltaX)**0.5
		if NDelta/N1 < tol: return (X,i)

		F1 = fun(X)
		Y = F1 - F0
		J += ((Y-J.dot(DeltaX)).dot(DeltaX.T))/(DeltaX.T.dot(DeltaX))
		F0 = F1

	X = np.array([[0.5*i] for i in range(size)])
	J = np.array([[0.5*i+0.5*j for i in range(size)] for j in range(size-1, -1, -1)])

	raise ValueError("broydenMultiEq() method with starting vector X:\n"+str(X)+"\n\nand J:\n"+str(J)+"\n\ndid not reach convergence.")



def nonLinearLSQ(function, n, X, Y, tol = 0.0001, reps = 1000):
	B = np.ones((X.size,n))
	symbolFunctions = sp.Matrix(sp.sympify(functions))
	symbolList = sp.symbols("x1:%d"%(n+1))
	fun = sp.lambdify([symbolList], symbolFunctions)

	for i in range(reps):

functions = ["x1 + x2 - 1", "x1 - x2 + 1"]
fun = sp.Matrix(sp.sympify(functions))
l = sp.symbols("x1 x2")
f = sp.lambdify([l], fun)
v = np.array([1,2], dtype = float)

print(derivative(f,v))
print(richardExtrapol(f,v))


startTime = time.time()
(X,i) = broydenMultiEq(functions)
endTime = time.time()

print("\nX:\n"+str(X))
print("\n# of reps = "+str(i))
print("\nTime = "+str(endTime-startTime))