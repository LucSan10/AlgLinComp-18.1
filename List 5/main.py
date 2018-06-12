# -*- coding: utf-8 -*-

import math
import matplotlib.pyplot as plt
from Funcao import Funcao
import os

def h(x):
    return math.exp(-0.5*x*x)/(2*math.pi)**0.5

func = Funcao(h)
print("f(x) = e**(-0.5*x**2)/sqrt(2pi)\n")
print("Integral 0-1:")
print("Quadratura de Gauss: "+str(func.integracao_quadratura(0,1,7))+"\n")

print("Integral 0-5:")
print("Quadratura de Gauss: "+str(func.integracao_quadratura(0,5,7))+"\n")

input()
os.system('clear')

def edo(t,y):
    return -2*t*y*y

func = Funcao(edo)
euler = func.Euler(0,2,1)
rk2 = func.Runge_Kutta2(0,2,1)
rk4 = func.Runge_Kutta4(0,2,1)

def solexata(t):
    return 1/(1+t*t)

t = [i*0.1 for i in range(21)]
plt.plot(t,euler,'ro')
plt.ylabel('Euler')
plt.show()

plt.plot(t,rk2,'ro')
plt.ylabel('R.K.2')
plt.show()

plt.plot(t,rk4,'ro')
plt.ylabel('R.K.4')
plt.show()

input()
os.system('clear')

def F(t):
    return 2*math.sin(0.5*t) + math.sin(2*0.5*t) + math.cos(3*0.5*t)

def edo2(t,y,dy):
    return F(t) - 0.2*dy - y

func = Funcao(edo2)
taylor = func.Taylor(0,100,0,0)
rkn = func.Runge_Kutta_Nystrom(0,100,0,0)

t = [0.1*i for i in range(1001)]

plt.plot(t,taylor,'ro')
plt.ylabel('Taylor')
plt.show()

plt.plot(t,rkn,'ro')
plt.ylabel('R.K.N')
plt.show()

print("Fim do programa.")