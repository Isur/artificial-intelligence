# -*- coding: utf-8 -*-
"""
Created on Sun May 13 11:19:21 2018

@author: Isur
"""

# Import:
import neurolab as nl
import numpy as np
import pylab as pl
from sklearn.datasets.mldata import fetch_mldata
from sklearn.datasets import load_diabetes
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import MLPClassifier
import scipy.io as sio
from sklearn import svm


import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
# Zadanie 1:

print("f(x) = sin(x)")
print("0 < x < 6")
x = np.linspace(0, 6, 20)
y = np.sin(x)
size = len(x)

inp = x.reshape(size,1)
tar = y.reshape(size,1)
net = nl.net.newff([[0, 6]],[5, 1])
net.trainf = nl.train.train_gd
error = net.train(inp, tar, epochs=500, show=100, goal=0.002)
 
out = net.sim(inp)

x2 = np.linspace(0,6,150)
y2 = net.sim(x2.reshape(x2.size,1)).reshape(x2.size)
y3 = net.sim(inp).reshape(size)
 
pl.plot(x2, y2, '-',x , y, '.', x, y3, 'p')
pl.legend(['wartosc rzeczywista', 'wynik uczenia'])
pl.show()

print("f(x) = 0.5log(x)")
print("1 < x < 2.5")
x = np.linspace(1, 2.5, 20)
y = np.log(x)*(1/2)
size = len(x)

inp = x.reshape(size,1)
tar = y.reshape(size,1)
net = nl.net.newff([[1, 2.5]],[5, 1])
net.trainf = nl.train.train_gd
error = net.train(inp, tar, epochs=500, show=100, goal=0.002)
 
out = net.sim(inp)

x2 = np.linspace(1,2.5,150)
y2 = net.sim(x2.reshape(x2.size,1)).reshape(x2.size)
y3 = net.sim(inp).reshape(size)
 
pl.plot(x2, y2, '-',x , y, '.', x, y3, 'p')
pl.legend(['wartosc rzeczywista', 'wynik uczenia'])
pl.show()

print("f(x) = xcos(x) + 0.3log(x)")
print("1 < x < 6")
x = np.linspace(1, 6, 20)
y = np.cos(x)*x + 0.3*np.log(x)
size = len(x)

inp = x.reshape(size,1)
tar = y.reshape(size,1)
net = nl.net.newff([[1, 6]],[5, 1])
net.trainf = nl.train.train_gd
error = net.train(inp, tar, epochs=500, show=100, goal=0.002)
 
out = net.sim(inp)

x2 = np.linspace(1,6,150)
y2 = net.sim(x2.reshape(x2.size,1)).reshape(x2.size)
y3 = net.sim(inp).reshape(size)
 
pl.plot(x2, y2, '-',x , y, '.', x, y3, 'p')
pl.legend(['wartosc rzeczywista', 'wynik uczenia'])
pl.show()

# Zadanie 1:
data = load_iris()
train, test, train_targets, test_targets = train_test_split(data.data, data.target, test_size=0.50, random_state = 42)
print("55,55,relu")
for i in range(0,5):
    p = MLPClassifier(hidden_layer_sizes=(55,55), activation="relu", max_iter=1000)
    p.fit(train, np.ravel(train_targets))
    print('błąd: ', 1 - p.score(test, np.ravel(test_targets)))
print("80,80,identity")
for i in range(0,5):
    p = MLPClassifier(hidden_layer_sizes=(80,80), max_iter=1000, activation="identity")
    p.fit(train, np.ravel(train_targets))
    print('błąd: ', 1 - p.score(test, np.ravel(test_targets)))
print("100,100,tanh")
for i in range(0,5):
    p = MLPClassifier(hidden_layer_sizes=(100,100), max_iter=1000, activation='tanh')
    p.fit(train, np.ravel(train_targets))
    print('błąd: ', 1 - p.score(test, np.ravel(test_targets)))

# Zadanie 2:
x = np.linspace(1, 2.5, 20)
y = np.log(x)*(1/2)
size = len(x)

inp = x.reshape(size,1)
tar = y.reshape(size,1)

net1 = nl.net.newff([[1, 2.5]],[5, 1])
net2 = nl.net.newff([[-1, 2.5]],[5, 1])
net3 = nl.net.newff([[-2.5, 5]],[5, 1])
net4 = nl.net.newff([[-10, 10]],[5, 1])
net5 = nl.net.newff([[7, 15]],[5, 1])

print("Wartosci: 1, 2.5")
net1.trainf = nl.train.train_gd
error1 = net1.train(inp, tar, epochs=500, show=100, goal=0.002)
out1 = net1.sim(inp)
x2 = np.linspace(1,2.5,150)
y2 = net1.sim(x2.reshape(x2.size,1)).reshape(x2.size)
y3 = net1.sim(inp).reshape(size)
 
pl.plot(x2, y2, '-',x , y, '.', x, y3, 'p')
pl.legend(['wartosc rzeczywista', 'wynik uczenia'])
pl.show()
print("Wartosci: -1, 2.5")
net2.trainf = nl.train.train_gd
error2 = net2.train(inp, tar, epochs=500, show=100, goal=0.002)
out2 = net2.sim(inp)
x2 = np.linspace(1,2.5,150)
y2 = net2.sim(x2.reshape(x2.size,1)).reshape(x2.size)
y3 = net2.sim(inp).reshape(size)
 
pl.plot(x2, y2, '-',x , y, '.', x, y3, 'p')
pl.legend(['wartosc rzeczywista', 'wynik uczenia'])
pl.show()
print("Wartosci: -2.5, 5")
net3.trainf = nl.train.train_gd
error3 = net3.train(inp, tar, epochs=500, show=100, goal=0.002)
out3 = net3.sim(inp)
x2 = np.linspace(1,2.5,150)
y2 = net3.sim(x2.reshape(x2.size,1)).reshape(x2.size)
y3 = net3.sim(inp).reshape(size)
 
pl.plot(x2, y2, '-',x , y, '.', x, y3, 'p')
pl.legend(['wartosc rzeczywista', 'wynik uczenia'])
pl.show()
print("Wartosci: -10, 10")
net4.trainf = nl.train.train_gd
error4 = net4.train(inp, tar, epochs=500, show=100, goal=0.002)
out4 = net4.sim(inp)
x2 = np.linspace(1,2.5,150)
y2 = net4.sim(x2.reshape(x2.size,1)).reshape(x2.size)
y3 = net4.sim(inp).reshape(size)
 
pl.plot(x2, y2, '-',x , y, '.', x, y3, 'p')
pl.legend(['wartosc rzeczywista', 'wynik uczenia'])
pl.show()
print("Wartosci: 7, 15")
net5.trainf = nl.train.train_gd
error5 = net5.train(inp, tar, epochs=500, show=100, goal=0.002)
out5 = net5.sim(inp)
x2 = np.linspace(1,2.5,150)
y2 = net5.sim(x2.reshape(x2.size,1)).reshape(x2.size)
y3 = net5.sim(inp).reshape(size)
 
pl.plot(x2, y2, '-',x , y, '.', x, y3, 'p')
pl.legend(['wartosc rzeczywista', 'wynik uczenia'])
pl.show()

# Zadanie 3:

data = fetch_mldata('MNIST')
train, test, train_targets, test_targets = train_test_split(data.data, data.target, test_size=0.5, random_state = 42)

mlp = MLPClassifier(solver='adam', alpha=0.0001)
mlp.fit(train, train_targets).predict(test)
print(mlp.score(test,test_targets))

mlp = MLPClassifier(solver='lbfgs', alpha=0.0001)
mlp.fit(train, train_targets).predict(test)
print(mlp.score(test,test_targets))

mlp = MLPClassifier(alpha=0.000001)
mlp.fit(train, train_targets).predict(test)
print(mlp.score(test,test_targets))

mlp = MLPClassifier(alpha=0.00001)
mlp.fit(train, train_targets).predict(test)
print(mlp.score(test,test_targets))

mlp = MLPClassifier(alpha=0.00001, solver='lbfgs')
mlp.fit(train, train_targets).predict(test)
print(mlp.score(test,test_targets))

mlp = MLPClassifier(alpha=0.000001, solver='lbfgs')
mlp.fit(train, train_targets).predict(test)
print(mlp.score(test,test_targets))
# Zadanie 4:
data = sio.loadmat('mnist_012.mat')
image_train = data['train_images']
label_train = data['train_labels']
image_test = data['test_images']
label_test = data['test_labels']
 
y, x, n = image_train.shape
image_train = image_train.reshape((n, x*y))
 
y,x, n = image_test.shape
image_test = image_test.reshape((n, x*y))

def mlpCls(x):
    mlp = MLPClassifier(hidden_layer_sizes=(x)).fit(image_train, label_train.ravel(n))
    f = nl.error.MSE()
    g = np.array(mlp.predict(image_test))
    return f(g, 0)

maxRes = 0
maxResIndex = 0
for i in range(1,100):
    res = mlpCls(i)
    if maxResIndex == 0:
        maxRes = res
        maxResIndex = i
    if maxRes > res:
        maxRes = res
        maxResIndex = i
print("Min MSE Error for: ", maxResIndex, " neurons is: ", maxRes)

best_mlp = MLPClassifier(hidden_layer_sizes=(maxResIndex)).fit(image_train, label_train.ravel(n))

print("Best score for ", maxResIndex, " neurons is: ", best_mlp.score(image_test, label_test))

# Zadanie 5:
data_1 = sio.loadmat('perceptron1.mat')
data_2 = sio.loadmat('perceptron2.mat')

perceptron_1_data = data_1['data']
perceptron_1_label = data_1['labels']
perceptron_2_data = data_2['data']
perceptron_2_label = data_2['labels']

net = nl.net.newp([[-10,10],[-5,5]],1)
print("Perceptron 1")
error1 = net.train(perceptron_1_data, perceptron_1_label, epochs=10, show=1, lr=0.01)
print("Perceptron 2")
error2 = net.train(perceptron_1_data, perceptron_1_label, epochs=10, show=1, lr=0.01)



# Zadanie 6:
data = load_diabetes()
train, test, train_targets, test_targets = train_test_split(data.data, data.target, test_size=0.5, random_state = 42)
train_targets = train_targets.reshape(len(train),1)
print("Perceptron")
net = nl.net.newp([[-5,5],[-5,5],[-5,5],[-5,5],[-5,5],[-5,5],[-5,5],[-5,5],[-5,5],[-5,5]],1)
error = net.train(train,train_targets)
print("MLP")
net = nl.net.newff([[-5,5],[-5,5],[-5,5],[-5,5],[-5,5],[-5,5],[-5,5],[-5,5],[-5,5],[-5,5]],[5,1])
error = net.train(train,train_targets)
print(error)
print("SVM")
print(svm.SVC().fit(train, train_targets.ravel()).predict(test))
# Zadanie 7:
data = sio.loadmat('banana.mat')
train = data['train_data']
train_label = data['train_labels']
test = data['test_data']
test_label = data['test_labels']

p = nl.net.newp([[-1,2],[0,2]],1)
p.train(train, train_label)
out = p.sim(test)
f = nl.error.MSE()
test_error = f(test_label, out)
print("Test error: ", test_error)


# Zadanie 8:
chars = ['a', 't', 'v']
target = np.asfarray(
        [[1,1,1,1,1,
          0,0,0,0,1,
          0,1,1,1,1,
          1,0,0,0,1,
          0,1,1,1,1],
         [0,0,1,0,0,
          1,1,1,1,1,
          0,0,1,0,0,
          0,0,1,0,0,
          0,0,1,1,0],
         [0,0,0,0,0,
          0,0,0,0,0,
          1,0,0,0,1,
          0,1,0,1,0,
          0,0,1,0,0]])
target[target==0] = -1
net = nl.net.newhop(target)
out = net.sim(target)
print("Test:")
for i in range(len(target)):
    print(chars[i],(out[i] == target[i]).all())
print("Defaced 'v' test:")
test = np.asfarray(
         [0,0,0,0,0,
          1,0,0,0,1,
          1,1,0,1,1,
          0,1,1,1,0,
          0,0,1,0,0,])
test[test==0] = -1
out = net.sim([test])
print((out[0]==target[2]).all(), " steps ", len(net.layers[0].outs))
# Zadanie 10:
with open('kohonen1.mat') as file:
    content = file.read()
fileContent = content.split('\n')
data = []
target = []
for x in fileContent:
    string = x.split(' ')
    try:
        data.append([float(string[1]), float(string[2])])
    except :
        try:
            target.append(int(string[1]))
        except:
            continue

net1 = nl.net.newc([[0,1],[1,5]],2)
net2 = nl.net.newc([[-5,15],[0,10]],3)
net3 = nl.net.newc([[10,15],[15,20]],5)
error = net1.train(data, epochs=200, show=40)
error2 = net2.train(data, epochs=400, show=80)
error3 = net3.train(data, epochs=800, show=160)
w = net1.layers[0].np['w']
w2 = net2.layers[0].np['w']
w3 = net3.layers[0].np['w']
pl.plot(data[0],data[1], w[:,0], w[:,1], 'p', w2[:,0], w2[:,1], 'p',w3[:,0], w3[:,1], 'p')
pl.legend(['data','net','net2','net3'])
pl.show()