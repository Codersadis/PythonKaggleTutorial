# -*- coding: utf-8 -*-

'''
python machine learning in Kaggle
Author: cmjia
'''

import pandas as pd

# read training data
df_train = pd.read_csv('../../Datasets/breast-cancer/breast-cancer-train.csv')

# read test data
df_test = pd.read_csv('../../Datasets/breast-cancer/breast-cancer-test.csv')

# feature collection & sampling data
df_test_negative = df_test.loc[df_test['Type']==0][['Clump Thickness', 'Cell Size']]
df_test_positive = df_test.loc[df_test['Type']==1][['Clump Thickness', 'Cell Size']]

import matplotlib.pyplot as plt

# plot negative samples with x while positive sammples with x
plt.scatter(df_test_negative['Clump Thickness'], df_test_negative['Cell Size'], marker='o', s=200, c='red')
plt.scatter(df_test_positive['Clump Thickness'], df_test_positive['Cell Size'], marker='x', s=150, c='black')

import numpy as np
intercept = np.random.random([1])
coef = np.random.random([2])
lx = np.arange(0,12)
ly = (- intercept - lx * coef[0]) / coef[1]

plt.plot(lx, ly, c='yellow')

# plot random variable classifier
plt.scatter(df_test_negative['Clump Thickness'], df_test_negative['Cell Size'], marker='o', s=200, c='red')
plt.scatter(df_test_positive['Clump Thickness'], df_test_positive['Cell Size'], marker='x', s=150, c='black')
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
plt.show()

# using first 10 training samples to fit 
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

lr.fit(df_train[['Clump Thickness', 'Cell Size']][:10], df_train['Type'][:10])
print('Testing accuracy (10 training samples):', lr.score(df_test[['Clump Thickness', 'Cell Size']], df_test['Type']))


'''
Using Logistic Regression, first 10 samples
'''
intercept = lr.intercept_
coef = lr.coef_[0, :]
ly = (-intercept - lx * coef[0]) / coef[1]
plt.plot(lx, ly, c='green')
plt.scatter(df_test_negative['Clump Thickness'], df_test_negative['Cell Size'], marker='o', s=200, c='red')
plt.scatter(df_test_positive['Clump Thickness'], df_test_positive['Cell Size'], marker='x', s=150, c='black')
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
plt.show()

'''
using all training samples
'''
lr = LogisticRegression()
lr.fit(df_train[['Clump Thickness', 'Cell Size']],df_train['Type'])
print('Testing accuracy (all training samples):', lr.score(df_test[['Clump Thickness', 'Cell Size']], df_test['Type']))

intercept = lr.intercept_
coef = lr.coef_[0, :]
ly  = (- intercept - lx * coef[0]) / coef[1]

plt.scatter
plt.plot(lx, ly, c='blue')
plt.scatter(df_test_negative['Clump Thickness'], df_test_negative['Cell Size'], marker='o', s=200, c='red')
plt.scatter(df_test_positive['Clump Thickness'], df_test_positive['Cell Size'], marker='x', s=150, c='black')
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
plt.show()










