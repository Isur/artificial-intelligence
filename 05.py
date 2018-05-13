# -*- coding: utf-8 -*-
"""
Created on Sun May 13 11:19:21 2018

@author: Isur
"""

# Import:
import neurolab as nl
import numpy as np
import pylab as pl

# Zadanie 1:

# f(x) = sin(x)
# 0 < x < 6

x = np.linspace(0, 6, 20)
y = np.sin(x)
size = len(x)

inp = x.reshape(size,1)
tar = y.reshape(size,1)
net = nl.net.newff([[0, 6]],[5, 1])
net.trainf = nl.train.train_gd
error = net.train(inp, tar, epochs=500, show=100, goal=0.02)
 
out = net.sim(inp)

x2 = np.linspace(0,6,150)
y2 = net.sim(x2.reshape(x2.size,1)).reshape(x2.size)
y3 = net.sim(inp).reshape(size)
 
pl.plot(x2, y2, '-',x , y, '.', x, y3, 'p')
pl.legend(['wartosc rzeczywista', 'wynik uczenia'])
pl.show()

# f(x) = 0.5log(x)
# 1 < x < 2.5

x = np.linspace(1, 2.5, 20)
y = np.log(x)*(1/2)
size = len(x)

inp = x.reshape(size,1)
tar = y.reshape(size,1)
net = nl.net.newff([[1, 2.5]],[5, 1])
net.trainf = nl.train.train_gd
error = net.train(inp, tar, epochs=500, show=100, goal=0.02)
 
out = net.sim(inp)

x2 = np.linspace(1,2.5,150)
y2 = net.sim(x2.reshape(x2.size,1)).reshape(x2.size)
y3 = net.sim(inp).reshape(size)
 
pl.plot(x2, y2, '-',x , y, '.', x, y3, 'p')
pl.legend(['wartosc rzeczywista', 'wynik uczenia'])
pl.show()

# f(x) = xcos(x) + 0.3log(x)
# 1 < x < 6

x = np.linspace(1, 6, 20)
y = np.cos(x)*x + 0.3*np.log(x)
size = len(x)

inp = x.reshape(size,1)
tar = y.reshape(size,1)
net = nl.net.newff([[1, 6]],[5, 1])
net.trainf = nl.train.train_gd
error = net.train(inp, tar, epochs=500, show=100, goal=0.02)
 
out = net.sim(inp)

x2 = np.linspace(1,6,150)
y2 = net.sim(x2.reshape(x2.size,1)).reshape(x2.size)
y3 = net.sim(inp).reshape(size)
 
pl.plot(x2, y2, '-',x , y, '.', x, y3, 'p')
pl.legend(['wartosc rzeczywista', 'wynik uczenia'])
pl.show()

# Zadanie 2:

# Zadanie 3:

# Zadanie 4:

# Zadanie 5:

# Zadanie 6:

# Zadanie 7:

# Zadanie 8:

# Zadanie 9:

# Zadanie 10:

# Zadanie 11:

