# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 12:33:43 2017

@author: cmjia
"""

from sklearn.datasets import load_boston
boston = load_boston()

# check dataset 
# print (boston)

from sklearn.cross_validation import train_test_split
import numpy as np

X = boston.data
y = boston.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 33,
                                                    test_size = 0.25)

print ('The max target value is', np.max(boston.target))
print ('The min target value is', np.min(boston.target))
print ('The average value target is', np.mean(boston.target))

# normalize data
from sklearn.preprocessing import StandardScaler
ss_X = StandardScaler()
ss_y = StandardScaler()

X_train = ss_X.fit_transform(X_train)
X_test = ss_X.fit_transform(X_test)
y_train = ss_y.fit_transform(y_train)
y_test = ss_X.fit_transform(y_test)

# linear regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_y_predict = lr.predict(X_test)

# SGD regression
from sklearn.linear_model import SGDRegressor
sgdr = SGDRegressor()

sgdr.fit(X_train, y_train)
sgdr_y_predict = sgdr.predict(X_test)

'''
Using THREE metrics to evaluate the regression performance
1. lr
2. sgdr
'''

# 1
print ('The value of default measurement of LinearRegression is', lr.score(X_test, y_test))
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
print ('The value of R-squared of LinearRegression is', r2_score(y_test,lr_y_predict))
print ('The mean square error of LinearRegression is', mean_squared_error(ss_y.inverse_transform(y_test), 
                                                                          ss_y.inverse_transform(lr_y_predict)))
print ('The mean absolute error of LinearRegression is', mean_absolute_error(ss_y.inverse_transform(y_test), 
                                                                             ss_y.inverse_transform(lr_y_predict)))

# 2
print ('The value of default measurement of SGDRegression is', sgdr.score(X_test, y_test))
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
print ('The value of R-squared of SGDRegression is', r2_score(y_test,sgdr_y_predict))
print ('The mean square error of SGDRegression is', mean_squared_error(ss_y.inverse_transform(y_test), 
                                                                          ss_y.inverse_transform(sgdr_y_predict)))
print ('The mean absolute error of SGDRegression is', mean_absolute_error(ss_y.inverse_transform(y_test), 
                                                                             ss_y.inverse_transform(sgdr_y_predict)))

