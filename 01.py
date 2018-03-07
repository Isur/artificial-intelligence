from sklearn import preprocessing
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import math
import random
file = open("iris.data")
# 01
data = []
for line in file:
    row = line.rstrip().split(',')
    data.append(row)
print("Number of samples: ", len(data))
print("Number of attributes: ", len(data[0]))
# 02
def printAttr(sample,y):
# y - sample number
    print("Attributes of sample ", y, " : ")
    for i in sample[y-1]:
        print(i)
    
printAttr(data, 10)
printAttr(data, 75)
eukllidesDistance = 0
for i in range(0,4):
    eukllidesDistance += (float(data[9][i])-float(data[74][i]))**2
eukllidesDistance = math.sqrt(eukllidesDistance)
print("Euklides distance = ",eukllidesDistance)
# 03
def analyze(attr):
    print("Minimmum: ", np.amin(attr,axis=0))
    print("Maximum: ", np.amax(attr,axis=0))
    print("Average: ", np.mean(attr,axis=0))
    print("Standard deviation: ", np.std(attr,axis=0))

attr_one = []
attr_two = []
attr_three = []
attr_four = []
attr_label = []
attr = []
attr.append(attr_one)
attr.append(attr_two)
attr.append(attr_three)
attr.append(attr_four)
attr.append(attr_label)
for i in data:
    attr_one.append(float(i[0]))
    attr_two.append(float(i[1]))
    attr_three.append(float(i[2]))
    attr_four.append(float(i[3]))
    attr_label.append(i[4])
for i in range(0,4):
    print("Analyze of ", i+1, " attribute: ")
    analyze(attr[i])
# 04
plt.scatter(attr[0],attr[1],c="RED")
plt.show()
# 05
setosaX = []
setosaY = []
versicolorX = []
versicolorY = []
virginicaX = []
virginicaY = []
for i in data:
    if i[4] == "Iris-setosa":
        setosaX.append(float(i[0]))
        setosaY.append(float(i[2]))
    elif i[4] == "Iris-versicolor":
        versicolorX.append(float(i[0]))
        versicolorY.append(float(i[2]))
    elif i[4] == "Iris-virginica":
        virginicaX.append(float(i[0]))
        virginicaY.append(float(i[2]))
plt.scatter(setosaX,setosaY, c="RED")
plt.scatter(versicolorX,versicolorY, c="BLUE")
plt.scatter(virginicaX,virginicaY, c="GREEN")
plt.show()
# 06
sum_setosa = [0,0,0,0]
counter_setosa = 0
sum_versicolor = [0,0,0,0]
counter_versicolor = 0
for i in data:
    if i[4] == "Iris-setosa":
        counter_setosa += 1
        for x in range(0,4):
            sum_setosa[x] += float(i[x])
    elif i[4] == "Iris-versicolor":
        counter_versicolor += 1
        for x in range(0,4):
            sum_versicolor[x] += float(i[x])
for i in range(0,4):
    avg = sum_setosa[i]/counter_setosa
    print("Average of setosa attribute    : ",i+1," = ", avg )
    avg = sum_versicolor[i]/counter_versicolor 
    print("Average of versicolor attribute: ",i+1," = ", avg )
# 07
n_attr_one = []
n_attr_two = []
n_attr_three = []
n_attr_four = []
n_attr = []
n_attr.append(n_attr_one)
n_attr.append(n_attr_two)
n_attr.append(n_attr_three)
n_attr.append(n_attr_four)
 
for i in range(0,4):
    for x in attr[i]:
        n_attr[i].append((x-np.average(attr[i],axis=0))/np.std(attr[i],axis=0))
for i in range(0,4):
    print("Minimum ", i+1, ": ", np.amin(n_attr[i],axis=0))
    print("Maximum ", i+1, ": ", np.amax(n_attr[i],axis=0))
    print("Average ", i+1, ": ", np.average(n_attr[i],axis=0))
    print("STD     ", i+1, ": ", np.std(n_attr[i],axis=0))
# 08
howMany = 10
x1 = np.random.randn(howMany) - 2
x2 = 11*np.random.rand(howMany)
data_rand = np.vstack((x1,x2))
data_rand = data_rand.conj().transpose()
plt.scatter(data_rand[:,0], data_rand[:,1])
plt.show()

# 09
euk = metrics.pairwise.pairwise_distances(data_rand, metric='euclidean')
print("Euklides Matrix: \n", euk)
mah = metrics.pairwise.pairwise_distances(data_rand, metric='mahalanobis')
print("Mahalanobis Matrix: \n", mah)
mink = metrics.pairwise.pairwise_distances(data_rand, metric='minkowski')
print("Minkowski Matrix: \n", mink)

# 10
scale = preprocessing.MinMaxScaler((0,1))
data_scale = scale.fit_transform(data_rand)
euk= metrics.pairwise.pairwise_distances(data_scale, metric='euclidean')
print("Euklides Matrix scale: \n", euk)
mah = metrics.pairwise.pairwise_distances(data_scale, metric='mahalanobis')
print("Mahalanobis Matrix scale: \n", mah)
mink = metrics.pairwise.pairwise_distances(data_scale, metric='minkowski')
print("Minkowski Matrix scale: \n", mink)

# 11
def d1(x):
    return -x[0]+x[1]
def d2(x):
    return x[0]-x[1]
def classify(x):
    if(d1(x)>d2(x)):
        return True
    else:
        return False
#  Wzór powierzchni decyzyjnej :
#  x1 = x2
# 12

howMany = 10
example_data = []
for i in range(0,howMany):
    example_data.append([random.uniform(-5,0),random.uniform(0,5)])
    example_data.append([random.uniform(0,5),random.uniform(-5,0)])
class_one_x = []
class_one_y = []
class_two_x = []
class_two_y = []
for i in example_data:
    if classify(i):
        class_one_x.append(i[0])
        class_one_y.append(i[1])
    else:
        class_two_x.append(i[0])
        class_two_y.append(i[1])

plt.scatter(class_one_x, class_one_y, c="GREEN")
plt.scatter(class_two_x, class_two_y, c="BLUE")
plt.plot([-5,5],[0,0])
plt.plot([0,0],[-5,5])
plt.show()
