"""
Simple decision tree.
"""
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.externals.six import StringIO
import pydotplus
import numpy as np

features = [[140, 1], [130, 1], [150, 0], [170, 0]]
labels = [0, 0, 1, 1]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

print clf.predict([[150, 0]])

iris = load_iris()
test_idx = [0, 50, 100]

#training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

#test data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print clf.predict(test_data)
print test_target

#Visualize the decision tree
dot_data = StringIO()
tree.export_graphviz(clf,
                     out_file=dot_data,
                     feature_names=iris.feature_names,
                     class_names=iris.target_names,
                     filled=True,
                     rounded=True,
                     special_characters=False)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf('decision_tree.pdf')
