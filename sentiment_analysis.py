#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn import naive_bayes
<<<<<<< HEAD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

=======
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
>>>>>>> 2d044970904bebc0eb6752df1a36399161d84d63
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold



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
vectorizer = CountVectorizer(stop_words=stopset)
X = vectorizer.fit_transform(dataframe.text)
X1 = vectorizer.fit_transform(df1.text)
X2 = vectorizer.fit_transform(df2.text)
y = dataframe.classy
y1 = df1.classy
y2 = df2.classy

# model = naive_bayes.MultinomialNB()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)


kf = KFold(n_splits=10)
print(kf)
for train_index, test_index in kf.split(X1):
    model = naive_bayes.MultinomialNB()

    print(X1[train_index][1:], y1[train_index][1:])
    model = model.fit(X1[train_index][1:], y1[train_index][1:])
    print(model)
    pred = model.predict(X1[test_index][1:])
    actual = np.array(y1[test_index][1:])
    # for i,j in zip(pred, actual):
    #     print(i,j)
    neutral_count=0
    positive_count=0
    negative_count=0
    neu_total=0
    p_total=0
    neg_total=0
    for i in range (len(actual)):
        p = pred[i]
        a = actual[i]
        # print(p,a)

    ##count totals for each case
        if a == '0':
            neu_total+=1
        elif a == '1':
            p_total+=1
        elif a == '-1':
            neg_total+=1

    ##count total predicted correctly for each case
        if p == '0' and p == a:
            neutral_count = neutral_count + 1
        if p == '1' and p == a:
            positive_count = positive_count + 1
        if p == '-1' and p == a:
            negative_count = negative_count + 1

    print("negative=", negative_count/neg_total)
    print("neutral=", neutral_count/neu_total)
    print("positive", positive_count/p_total)


model2 = naive_bayes.MultinomialNB()
model2 = model2.fit(X_train, y_train)
pred = model2.predict(X_test)
actual = np.array(y_test)
print(actual)
# accuracy = cross_val_score(model, X1, y1, cv=10, scoring='accuracy')
# print("cv=10 accuracy = ", accuracy)
# recall = cross_val_score(model, X, y, cv=10, scoring='recall')


# labels=[-1,0,1]
# print(f1_score(pred, actual, average='micro', labels=labels))
# f1 = cross_val_score(model, X, y, cv=10, scoring='f1')
#produces same results as F1_score method but doesn't have "elementwise comparison warning."
# f1score = model.score(X_test, y_test)
# print("F1_score = ",f1score)


neutral_count=0
positive_count=0
negative_count=0
neu_total=0
p_total=0
neg_total=0
precision_neu_total=0
precision_p_total=0
precision_neg_total=0

for i in range(len(actual)):
    p = pred[i]
    print(p)
    a = actual[i]
    # print(p,a)

##count totals for each case
    if a == '0':
        neu_total+=1
    elif a == '1':
        p_total+=1
    elif a == '-1':
        neg_total+=1

    if p == '0':
        precision_neu_total+=1
    elif p == '1':
        precision_p_total+=1
    elif p == '-1':
        precision_neg_total+=1


##count total predicted correctly for each case
    if p == '0' and p == a:
        neutral_count = neutral_count + 1
    if p == '1' and p == a:
        positive_count = positive_count + 1
    if p == '-1' and p == a:
        negative_count = negative_count + 1


print("recall_negative=", negative_count/neg_total)
print("recall_neutral=", neutral_count/neu_total)
print("recall_positive=", positive_count/p_total)

print("precision_negative=", negative_count/precision_neg_total)
print("precision_neutral=", neutral_count/precision_neu_total)
print("precision_positive=", positive_count/precision_p_total)

print("F-score_negative=", 2*((negative_count/precision_neg_total)*(negative_count/neg_total))/((negative_count/precision_neg_total)+(negative_count/neg_total)))
print("F-score_neutral=", 2*((neutral_count/precision_neu_total)*(neutral_count/neu_total))/((neutral_count/precision_neu_total)+(neutral_count/neu_total)))
print("F-score_positive=", 2*((positive_count/precision_p_total)*(positive_count/p_total))/((positive_count/precision_p_total)+(positive_count/p_total)))


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
