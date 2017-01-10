# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 10:17:24 2017

@author: cmjia
"""
# this dataset will be download online
from sklearn.datasets import fetch_20newsgroups
news = fetch_20newsgroups(subset='all')

# check dataset size
# print (len(news.data))
# print (news.data[0])

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(news.data, news.target, test_size = 0.25, random_state = 33)

# feature extraction
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
X_train = vec.fit_transform(X_train)
X_test = vec.transform(X_test)


# load Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
y_predict = mnb.predict(X_test)

from sklearn.metrics import classification_report

print ('The Accuracy of Naive Bayes Classifier is', mnb.score(X_test, y_test))
print (classification_report(y_test, y_predict, target_names=news.target_names))


