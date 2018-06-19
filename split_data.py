#!/usr/bin/python3
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
#loading data
iris=load_iris()
#training flower features stored in iris.data
#output accordingly stored in iris.target
#now splitting into test and train sets
train_iris,test_iris,train_target,test_target=train_test_split(iris.data,iris.target,test_size=0.2)
#now calling KNN 
knnclf=KNeighborsClassifier(n_neighbors=5)
#now calling Decision tree classifier
dsclf=tree.DecisionTreeClassifier()
#training data for KNN
trainedknn=knnclf.fit(train_iris,train_target)
#training data for DSC
traineddsc=dsclf.fit(train_iris,train_target)
#testing for KNN
outputknn=trainedknn.predict(test_iris)
print(outputknn)
#calculating accuracy for KNN
pctknn=accuracy_score(test_target,outputknn)
print(pctknn)
#testing for DSC
outputdsc=traineddsc.predict(test_iris)
print(outputdsc)
#calculating accuracy for DSC
pctdsc=accuracy_score(test_target,outputdsc)
print(pctdsc)
#original output
print(test_target)
a=tree.export_graphviz
print(a)
export_graphviz(dsclf, out_file="tree.dot", max_depth=7, feature_names=iris.feature_names, class_names=None, label='all', filled=True, leaves_parallel=False, impurity=True, node_ids=False, proportion=False, rounded=False)

plt.bar(outputknn,outputdsc,label="output")
plt.bar(pctknn,pctdsc,label="percent")
plt.show()
plt.legend()
