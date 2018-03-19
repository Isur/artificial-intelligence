import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import random
from sklearn import preprocessing
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
from matplotlib.colors import ListedColormap
from sklearn.cross_validation import train_test_split
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn import neighbors, datasets

# Zadanie 1:
banana_data = sio.loadmat('banana.mat')['test_data']
banana_labels = sio.loadmat('banana.mat')['test_labels']

train, test, train_targets, test_targets = train_test_split(
    banana_data, banana_labels, test_size=0.30, random_state=42)


# Zadanie 2:
gnb = GaussianNB()
predicted = gnb.fit(train, np.ravel(train_targets)).predict(test)

# Zadanie 3:
x = []
y = []
z = []
for i in range(len(test)):
    x.append(test[i][0])
    y.append(test[i][1])
    z.append(predicted[i])

for i in range(len(x)):
    if z[i] == 1:
        plt.scatter(x[i],y[i],c="RED")
    else:
        plt.scatter(x[i],y[i],c="GREEN")
        
        
c = 1.0
h = .02
x_min, x_max = test[:,0].min()-1,test[:,0].max()+1
y_min, y_max = test[:,1].min()-1,test[:,1].max()+1

xx,yy = np.meshgrid(np.arange(x_min, x_max,h), np.arange(y_min,y_max,h))

Z = gnb.predict(np.c_[xx.ravel(),yy.ravel()])
Z=Z.reshape(xx.shape)

plt.contour(xx,yy,Z,cmap=plt.cm.Paired)

plt.show()
# Zadanie 4:
print("Classifier efficiency: %f" % gnb.score(train,np.ravel(train_targets)))
# Zadanie 5:
clf = NearestCentroid()

# Zadanie 6:
clf.fit(train,np.ravel(train_targets))
predicted = clf.predict(test)
# Zadanie 7:
x = []
y = []
z = []
c1 = [0,0]
c2 = [0,0]
for i in range(len(test)):
    x.append(test[i][0])
    y.append(test[i][1])
    z.append(predicted[i])
    if(z[i]==1):
        c1[0]+=x[i]
        c1[1]+=y[i]
    else:
        c2[0]+=x[i]
        c2[1]+=y[i]

counter1 = 0
counter2 = 0
for i in range(len(x)):
    if z[i] == 1:
        plt.scatter(x[i],y[i],c="RED")
        counter1 += 1
    else:
        plt.scatter(x[i],y[i],c="GREEN")
        counter2 += 1
c1 = [c1[0]/counter1,c1[1]/counter1]
c2 = [c2[0]/counter2,c2[1]/counter2]
plt.scatter(c1[0],c1[1],c="BLUE")
plt.scatter(c2[0],c2[1],c="BROWN")
plt.show()

# Zadanie 8:
print("Classifier efficiency: %f" % clf.score(train,np.ravel(train_targets)))

# Zadanie 9:
k_best = [0,0]
for k in range(5,17):
    knn = neighbors.KNeighborsClassifier(k,weights='uniform', metric='euclidean')
    knn.fit(train,np.ravel(train_targets))
    predicted = knn.predict(test)
    sc = knn.score(train,np.ravel(train_targets))
    if sc > k_best[1]:
        k_best = [k,sc]
print("Best efficiency in 5-16 is for k =", k_best[0], " --> efficiency : ", k_best[1])

# Zadanie 10:
knn = neighbors.KNeighborsClassifier(k_best[0],weights='uniform', metric='euclidean')
knn.fit(train,np.ravel(train_targets))
predicted = knn.predict(test)
sc = knn.score(train,np.ravel(train_targets))
x = []
y = []
z = []
for i in range(len(test)):
    x.append(test[i][0])
    y.append(test[i][1])
    z.append(predicted[i])

for i in range(len(x)):
    if z[i] == 1:
        plt.scatter(x[i],y[i],c="RED")
    else:
        plt.scatter(x[i],y[i],c="GREEN")
plt.show()

# Zadanie 11:
print("Bad classify: ", math.ceil(len(x)*(1-sc)))