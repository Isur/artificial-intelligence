from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.externals.six import StringIO
import pydotplus
import numpy as np
import math


# Zadanie 1:
data = load_iris()
train, test, train_targets, test_targets = train_test_split(data.data, data.target, test_size=0.50, random_state = 42)

# Zadanie 2:
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train,train_targets)

dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris_tree.pdf")

# Zadanie 3:

predicted = clf.predict(test)
sc = clf.score(test,np.ravel(test_targets))
print("Classification efficiency: ", sc)
print("Bad classify: ", math.ceil(len(train)*(1-sc)))

# Zadanie 4:
# Indeks Giniego

# Zadanie 5:
predicted = clf.predict(test)
sc = clf.score(test,np.ravel(test_targets))
print("Classification efficiency: ", sc)

# Zadanie 6:
clf = tree.DecisionTreeClassifier(max_depth=2)
clf = clf.fit(train,train_targets)

dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris_tree_max_depth_2.pdf")
predicted = clf.predict(test)
sc = clf.score(test,np.ravel(test_targets))
print("Classification efficiency for depth 2: ", sc)


clf = tree.DecisionTreeClassifier(max_depth=3)
clf = clf.fit(train,train_targets)

dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris_tree_max_depth_3.pdf")
predicted = clf.predict(test)
sc = clf.score(test,np.ravel(test_targets))
print("Classification efficiency for depth 3: ", sc)

# Zadanie 7:
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(train,train_targets)

dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris_tree_entropy.pdf")

# Zadanie 8:

predicted = clf.predict(test)
sc = clf.score(test,np.ravel(test_targets))
print("Classification efficiency for entropy method: ", sc)

# Zadanie 9:

clf = tree.DecisionTreeClassifier(criterion='entropy', min_samples_leaf=4, max_leaf_nodes=4)
clf = clf.fit(train,train_targets)

dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris_tree_entropy_min_samples_5_max_nodes_2.pdf")

predicted = clf.predict(test)
sc = clf.score(test,np.ravel(test_targets))
print("Classification efficiency for entropy method with min 5 samples and max 2 nodes: ", sc)

# Zadanie 10:
train, test, train_targets, test_targets = train_test_split(data.data[:,:2], data.target, test_size=0.50, random_state = 42)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(train,train_targets)

dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris_tree_entropy_first_two_attributes.pdf")

predicted = clf.predict(test)
sc = clf.score(test,np.ravel(test_targets))
print("Classification efficiency for first two attributes ", sc)

