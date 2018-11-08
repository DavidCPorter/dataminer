#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn import naive_bayes
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

#read csv and label columns since default adds a space before some names
df1 = pd.read_csv('training_data/data-1_train.csv', names=['example_id', 'text', 'aspect_term', 'term_location', 'classy'])
df2 = pd.read_csv('training_data/data-2_train.csv', names=['example_id', 'text', 'aspect_term', 'term_location', 'classy'])

#remove the first row since it's header duplication
df1 = df1.drop([0], axis=0)
df2 = df2.drop([0], axis=0)

dataframe = df1.append(df2)

# In[16]:
stopset = set(stopwords.words('english'))
stopset.add('[comma]')
y = dataframe.classy
print(len(y))
vectorizer = CountVectorizer(stop_words=stopset)
X = vectorizer.fit_transform(dataframe.text)
print(X.shape)

#split training vs test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)



model = naive_bayes.MultinomialNB()
model.fit(X_train, y_train)

#try model
pred = model.predict(X_test)

actual = np.array(y_test)

count = 0

for i in range (len(actual)):
    if pred[i] == actual [i]:
        count = count + 1


print("Naive Bayes accuracy:")
print(count/len(pred))
print("Naive Bayes precision, recall, fscore")
print(precision_recall_fscore_support(actual, pred, average='weighted'))


# svm try
svm_model = svm.SVC(gamma='scale')
svm_model.fit(X_train, y_train)
svmpred = svm_model.predict(X_test)
count = 0
for i in range(len(actual)):
    if svmpred[i] == actual[i]:
        count = count + 1
print("svm accuracy:")
print(count / len(svmpred))
print("svm precision, recall, fscore")
print(precision_recall_fscore_support(actual, svmpred, average='weighted'))
