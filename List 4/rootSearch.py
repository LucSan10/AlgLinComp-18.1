import numpy as np
import time

def derivative(function, x, delta = 0.0001):
	front = function(x+delta)
	back = function(x-delta)
	return (front-back)/(2*delta)

def bisecRoot(function, start, end, tol = 0.001):
	if function(start)*function(end) > 0:
		raise ValueError("Interval limits a and b have same signs.")
	
	root = abs(end+start)/2.0
	value = function(root)

	while value != 0 and abs(end-start)/2.0 > tol:
		if value > 0: end = root
		else: start = root
		root = abs(end+start)/2.0
		value = function(root)
	return root

def newtonRoot(function, x0, tol = 0.0001, reps = 1000):
	for i in range(reps):
		x1 = x0 - function(x0)/derivative(function,x0)
		
		if abs(x1-x0) < tol:
			return x1
		x0 = x1
	
	raise ValueError("newtonRoot() method with given x0 = "+str(x0)+" did not reach convergence.")

def secantRoot(function, x0, tol = 0.0001, reps = 1000, delta = 0.01):
	x1 = x0+delta
	lastVal = function(x0)
	x2 = 0
	
	for i in range(reps):
		currentVal = function(x1)
		x2 = x1 - (currentVal*(x1-x0))/(currentVal-lastVal)
		if abs(x2-x1) < tol: return x2
		
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
		if abs(x4-x0) < tol: return x4
		
		l[np.argmax(np.abs(l))] = x4
		l.sort()
		[x1,x2,x3] = l
		x0 = x4

	raise ValueError("inverseInterpolRoot() method with starting x1, x2, x3 = ["+str(x1)+", "+str(x2)+", "+str(x3)+"] did not reach convergence")

def f(x): return x**3-8*x**2-6*x-5

startTime = time.time()
print(inverseInterpolRoot(f,6,8,10))
endTime = time.time()
print(endTime-startTime)