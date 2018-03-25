from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.externals.six import StringIO
import pydotplus
import numpy as np


# Zadanie 1:
data = load_iris()
train, test, train_targets, test_targets = train_test_split(data.data, data.target, test_size=0.50, random_state = 42)

# Zadanie 2:
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train,train_targets)
predicted = clf.predict(train)

#dot_data = StringIO()
#tree.export_graphviz(clf, out_file=dot_data)
#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
#graph.write_pdf("iris_tree.pdf")

# Zadanie 3:

right = 0
x = 0
for i in train_targets:
    if i==predicted[x]:
        right += 1
    x += 1
print('Right Classify:')
print(right)
print('Efficiency:')
CR=float(right)/len(predicted)
print(CR)