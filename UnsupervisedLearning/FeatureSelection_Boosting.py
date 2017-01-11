# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 17:46:23 2017

@author: cmjia
"""

import pandas as pd
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')

# separate feature and label
y = titanic['survived']
X = titanic.drop(['row.names', 'name', 'survived'], axis=1)

X['age'].fillna(X['age'].mean(), inplace=True)
X.fillna('UNKNOWN', inplace=True)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 33)

from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
X_test = vec.transform(X_test.to_dict(orient='record'))

# check feature dimension
# print (len(vec.feature_names_))

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion='entropy')
dt.fit(X_train, y_train)

print ('The performance of all features:', dt.score(X_test, y_test))
print ('')

from sklearn import feature_selection
fs = feature_selection.SelectPercentile(feature_selection.chi2, 
                                        percentile=20)
X_train_fs = fs.fit_transform(X_train, y_train)
dt.fit(X_train_fs, y_train)
X_test_fs = fs.transform(X_test)

print ('The performance of selected features:', dt.score(X_test_fs, y_test))
print ('')

from sklearn.cross_validation import cross_val_score
import numpy as np

percentiles = range(1, 100, 2)
results = []

for i in percentiles:
    fs = feature_selection.SelectPercentile(feature_selection.chi2,
                                            percentile=i)
    X_train_fs = fs.fit_transform(X_train, y_train)
    scores     = cross_val_score(dt, X_train_fs, y_train, cv=5)
    results    = np.append(results, scores.mean())

print ('The results in differenct percent features is:\n', results)
print ('')

# find the best percentage
opt = np.where(results == results.max())[0]
print ('The Optimal Number of Features is:', opt)
print ('')

# plot the best performance features
import matplotlib.pylab as plt
plt.plot(percentiles, results)
plt.xlabel('percentiles of features')
plt.ylabel('accuracy')
plt.show()

from sklearn import feature_selection
fs = feature_selection.SelectPercentile(feature_selection.chi2,
                                        percentile=7)
X_train_fs = fs.fit_transform(X_train, y_train)
dt.fit(X_train_fs, y_train)
X_test_fs = fs.transform(X_test)
print (dt.score(X_test_fs, y_test))
