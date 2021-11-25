import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import export_graphviz
import pydot

features = pd.read_csv('treeexample.csv')
#print(features.head(14))
features = pd.get_dummies(features)
#print(features.head(14))
labels = np.array(features['Default?'])
#print(labels)
features = features.drop('Default?', axis = 1)
features_list = list(features.columns)
#print(features_list)
features = np.array(features)
train_features = features[0:11]
test_features = features[11:]
train_labels = labels[0:11]
test_labels = labels[11:]
rf = RandomForestRegressor(n_estimators = 1000)
rf.fit(train_features, train_labels)
predictions = rf.predict(test_features)
print(predictions)
print(test_labels)
tree = rf.estimators_[5]
export_graphviz(tree, out_file = 'tree.dot', feature_names = features_list, rounded = True, precision = 1, filled = True)
(graph, ) = pydot.graph_from_dot_file('tree.dot')
graph.write_png('tree.png')
