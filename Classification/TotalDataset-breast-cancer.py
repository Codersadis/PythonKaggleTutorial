# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 15:31:52 2016

@author: cmjia
"""

import pandas as pd
import numpy as np

# define data attribute
column_names = ['Sample Code Number', 'Clump Thickness', 'Uniformity of Cell Size', 
'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
'Normal Nucleoli', 'Mitoses', 'Class']


data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',
                   names=column_names)
data = data.replace(to_replace='?', value=np.nan)
data = data.dropna(how='any') # discard all missing-attribute samples

### print dataset size
print('dataset size:', data.shape)

'''
split data into train/test sets
'''
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data[column_names[1:10]], 
                                                    data[column_names[10]], 
                                                    test_size=0.25, random_state=33)

# check train/test data size
print ('training set size:', y_train.value_counts())
print ('test set size:', y_test.value_counts())

'''
Linear Classifier for Prediction
'''
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)

lr = LogisticRegression()
sgdc = SGDClassifier()

# using LR fitting training samples
lr.fit(X_train, y_train) 
# inference 
lr_y_predict = lr.predict(X_test) 

# training process
sgdc.fit(X_train, y_train)
sgdc_y_predict = sgdc.predict(X_test)

'''
Performance Analysis
'''
from sklearn.metrics import classification_report
# LR
print ('Accuracy of LR Classifier:', lr.score(X_test, y_test))
print (classification_report(y_test, lr_y_predict, target_names=['Benign','Malignant']))

# SGD
print ('Accuracy of SGD Classifier:', sgdc.score(X_test, y_test))
print (classification_report(y_test, sgdc_y_predict, target_names=['Benign','Malignant']))












