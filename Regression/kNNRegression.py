# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 16:26:44 2017

@author: cmjia
"""

from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split

boston = load_boston()
X = boston.data
y = boston.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 33,
                                                    test_size = 0.25)

# normalize data
from sklearn.preprocessing import StandardScaler
ss_X = StandardScaler()
ss_y = StandardScaler()

X_train = ss_X.fit_transform(X_train)
X_test = ss_X.fit_transform(X_test)
y_train = ss_y.fit_transform(y_train)
y_test = ss_X.fit_transform(y_test)

from sklearn.neighbors import KNeighborsRegressor
uni_knr = KNeighborsRegressor(weights='uniform')
uni_knr.fit(X_train, y_train)
uni_knr_y_predict = uni_knr.predict(X_test)

dis_knr = KNeighborsRegressor(weights='distance')
dis_knr.fit(X_train, y_train)
dis_knr_y_predict = dis_knr.predict(X_test)

# evaluate
from sklearn.metrics import mean_absolute_error, mean_squared_error
print ('R-squared value of uniform-weighted KNNRegressor', uni_knr.score(X_test, y_test))
print ('The mean square error of uniform-weighted KNNRegressor', mean_squared_error(ss_y.inverse_transform(y_test),
                                                                 ss_y.inverse_transform(uni_knr_y_predict)))
print ('The mean absolute error of uniform-weighted KNNRegressor', mean_absolute_error(ss_y.inverse_transform(y_test),
                                                                 ss_y.inverse_transform(uni_knr_y_predict)))
print ('')

print ('R-squared value of distance-weighted KNNRegressor', dis_knr.score(X_test, y_test))
print ('The mean square error of distance-weighted KNNRegressor', mean_squared_error(ss_y.inverse_transform(y_test),
                                                                 ss_y.inverse_transform(dis_knr_y_predict)))
print ('The mean absolute error of distance-weighted KNNRegressor', mean_absolute_error(ss_y.inverse_transform(y_test),
                                                                 ss_y.inverse_transform(dis_knr_y_predict)))
print ('')




