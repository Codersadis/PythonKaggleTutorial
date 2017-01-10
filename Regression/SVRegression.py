# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 16:12:37 2017

@author: cmjia
"""

from sklearn.svm import SVR
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

# using linear kernel for SVM
linear_svr = SVR(kernel='linear')
linear_svr.fit(X_train, y_train)
linear_svr_y_predict = linear_svr.predict(X_test)

# using polynomial kernel for SVM
poly_svr = SVR(kernel='poly')
poly_svr.fit(X_train, y_train)
poly_svr_y_predict = poly_svr.predict(X_test)

# using rbf kernel for SVM
rbf_svr = SVR(kernel='rbf')
rbf_svr.fit(X_train, y_train)
rbf_svr_y_predict = rbf_svr.predict(X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error
print ('The value of default measurement of LinearSVR is', linear_svr.score(X_test, y_test)) 
print ('The mean square error of LinearSVR is', mean_squared_error(ss_y.inverse_transform(y_test), 
                                                                          ss_y.inverse_transform(linear_svr_y_predict)))
print ('The mean absolute error of LinearSVR is', mean_absolute_error(ss_y.inverse_transform(y_test), 
                                                                             ss_y.inverse_transform(linear_svr_y_predict)))
print ('')
print ('The value of default measurement of polySVR is', poly_svr.score(X_test, y_test)) 
print ('The mean square error of polySVR is', mean_squared_error(ss_y.inverse_transform(y_test), 
                                                                          ss_y.inverse_transform(poly_svr_y_predict)))
print ('The mean absolute error of polySVR is', mean_absolute_error(ss_y.inverse_transform(y_test), 
                                                                             ss_y.inverse_transform(poly_svr_y_predict)))
print ('')
print ('The value of default measurement of rbfSVR is', rbf_svr.score(X_test, y_test)) 
print ('The mean square error of rbfSVR is', mean_squared_error(ss_y.inverse_transform(y_test), 
                                                                          ss_y.inverse_transform(rbf_svr_y_predict)))
print ('The mean absolute error of rbfSVR is', mean_absolute_error(ss_y.inverse_transform(y_test), 
                                                                             ss_y.inverse_transform(rbf_svr_y_predict)))
print ('')
