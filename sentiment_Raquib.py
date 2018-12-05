#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
from sklearn import naive_bayes, svm, tree
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction import DictVectorizer

if __name__ == '__main__':
        #read csv and label columns since default adds a space before some names
        df1 = pd.read_csv('training_data/data-1_train.csv', names=['example_id', 'text', 'aspect_term', 'term_location', 'classy'])
        df2 = pd.read_csv('training_data/data-2_train.csv', names=['example_id', 'text', 'aspect_term', 'term_location', 'classy'])

        #remove the first row since it's header duplication
        df1 = df1.drop([0], axis=0)
        df2 = df2.drop([0], axis=0)

        # working with only df1 now
        dataframe = df1

        # In[16]:
        stopset = set(stopwords.words('english'))
        stopset.add('[comma]')
        y = dataframe.classy
        print(len(y))

        vectorizer = CountVectorizer(stop_words=stopset)
        # vectorizer = TfidfVectorizer(stop_words=stopset)

        X = vectorizer.fit_transform(dataframe.text)
        # print(vectorizer.get_feature_names())
        # print(X.shape)
        scoring = ['precision_macro', 'recall_macro', 'f1_macro', 'accuracy']


        # test with POS_TAG around aspect term

        # tempdictlist=[]
        # for i in range(1, len(y)+1):
        #         index=dataframe.term_location.loc[i]
        #         index=index.split("--")
        #         start=int(index[0])
        #         end=int(index[1])
        #
        #         text=dataframe.text.loc[i]
        #         leftText=text[:start-1]
        #         leftText=leftText.replace('[comma]',' ')
        #         firstTokens=word_tokenize(leftText)
        #         firstTokens=[word for word in firstTokens if not word in stopset]
        #
        #         rightText = text[end:]
        #         rightText = rightText.replace('[comma]',' ')
        #         lastTokens = word_tokenize(rightText)
        #         lastTokens=[word for word in lastTokens if not word in stopset]
        #
        #         blanklist=firstTokens
        #
        #         aspect = dataframe.aspect_term.loc[i]
        #         aspectToken=word_tokenize(aspect)
        #
        #         if len(firstTokens)>3:
        #                 del blanklist[:len(blanklist)-3]
        #
        #         blanklist.extend(aspectToken)
        #
        #         if len(lastTokens)>3:
        #                 del lastTokens[3:]
        #
        #         blanklist.extend(lastTokens)
        #
        #         # x=dataframe.text.loc[i]
        #         # words=word_tokenize(x)
        #
        #         POS= pos_tag(blanklist)
        #
        #         POS=dict(POS)
        #         tempdictlist.append(POS)
        #
        # newvectorizer=DictVectorizer()
        # POS_X = newvectorizer.fit_transform(tempdictlist)
        # newX = TfidfTransformer().fit_transform(POS_X)
        # # print(newvectorizer.get_feature_names())
        # # print(POS_X.shape)
        # # print(dataframe.text)
        # # print(type(y))
        # X=newX



        # # test
        # corpus = ['This is the first document.',
        #           'This document is the second document.',
        #           'And this is the third one.',
        #           'Is this the first document' ]
        #
        # newvectorizer = TfidfVectorizer()
        # testingX= newvectorizer.fit_transform(corpus)
        # print(newvectorizer.get_feature_names())
        # print(testingX.toarray())



        # svm try
        # cross validation try
        print("SVM with cross validation")
        svm_model = svm.SVC(gamma='scale')
        scores = cross_validate(svm_model,X,y, scoring=scoring,
                                cv = 5, return_train_score = False)
        # print(sorted(scores.keys()))
        print(np.mean(scores['test_accuracy']))



        # NB
        # cross validation try
        print("NB with cross validation")
        NB = naive_bayes.MultinomialNB()
        scores = cross_validate(NB, X, y, scoring=scoring,
                                cv=5, return_train_score=False)
        # print(sorted(scores.keys()))
        print(np.mean(scores['test_accuracy']))



        # DT
        # cross validation try
        print("Decision tree with cross validation")
        DT = tree.DecisionTreeClassifier()
        scores = cross_validate(DT, X, y, scoring=scoring,
                                cv=5, return_train_score=False)
        # print(sorted(scores.keys()))
        print(np.mean(scores['test_accuracy']))




        # Random forest
        # cross validation try
        print("Random Forest with cross validation")
        RF = RandomForestClassifier()
        scores = cross_validate(RF, X, y, scoring=scoring,
                                cv=5, return_train_score=False)
        # print(sorted(scores.keys()))
        print(np.mean(scores['test_accuracy']))


        # Adaboost
        # cross validation try
        print("Adaboost with cross validation")
        Ada = AdaBoostClassifier()
        scores = cross_validate(Ada, X, y, scoring=scoring,
                                cv=5, return_train_score=False)
        # print(sorted(scores.keys()))
        print(np.mean(scores['test_accuracy']))
