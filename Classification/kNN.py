# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 10:58:03 2017

@author: cmjia
"""

# load iris dataset in sklearn
from sklearn.datasets import load_iris
iris = load_iris()

# check dataset size
#print (iris.data.shape)

# check dataset discription
#print (iris.DESCR)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target,
                                                    test_size = 0.25, random_state = 33)

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# kNN predict
knc = KNeighborsClassifier()
knc.fit(X_train, y_train)
y_predict = knc.predict(X_test)

print ('The Accuracy of kNN Classifier is', knc.score(X_test, y_test))
from sklearn.metrics import classification_report

print (classification_report(y_test, y_predict, target_names=iris.target_names))

