#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn import naive_bayes, svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras import backend as K
from DFS import DFS
from keras.utils import to_categorical
from nltk.stem import PorterStemmer
nltk.download('wordnet')




# this doesn't work but it doesn't matter anymore since it was only asked for the presentation for nb and they will provide their own evaluation during the demo. But we could use kfold for the writeup as a secondary evaluator to our model.
def nb_trainNtest_10fold():
    kf = KFold(n_splits=10)
    # print(kf)
    for train_index, test_index in kf.split(X1):
        model = naive_bayes.MultinomialNB()
        print(train_index, test_index)
        print("x1-> ", X1)
        # print(X1[train_index][1:], y1[train_index][1:])
        model = model.fit(X1[train_index][1:], y1[train_index][1:])
        # print(model)
        pred = model.predict(X1[test_index][1:])
        actual = np.array(y1[test_index][1:])
        print(pred,actual)

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
            print(p,a)

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

    print("Naive Bayes precision, recall, fscore")

    print(precision_recall_fscore_support(actual, pred, average='weighted'))


# this simply used as a measure of evaluation for our NB model by printing pos, neu, and neg results. need to improve preprocessing for low numbers in each category.
def nb_trainNtest():
    model2 = naive_bayes.MultinomialNB()
    model2 = model2.fit(X_train, y_train)
    pred = model2.predict(X_test)
    actual = np.array(y_test)


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
        # print(p)
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

    count = negative_count+positive_count+neutral_count
    print("Naive Bayes accuracy:")
    print(count/len(pred))
    print("Naive Bayes precision, recall, fscore")
    print(precision_recall_fscore_support(actual, pred, average='weighted'))


def nb_for_demo(demo_file):
     # train model per usual with previously provided data
    model2 = naive_bayes.MultinomialNB()
    model2 = model2.fit(X_train, y_train)
    # dont really need these two lines
    # print(len(X_test))
    pred = model2.predict(X_test)
    actual = np.array(y_test)


# **PREPROCESS DEMO FILE**
    df = pd.read_csv(demo_file, names=['example_id', 'text', 'aspect_term', 'term_location', 'classy'])
    # remove header
    df = df.drop([0], axis=0)
    # vectorizes the text column (bag of words)
    df_vectorized = vectorizer.transform(df.text)
    # get predictions
    demo_pred = model2.predict(df_vectorized)

    f = open("david_porter_raquib_yousuf.txt", "w")
    example_ids = df['example_id'].values
    print(len(demo_pred))
    for i in range(len(demo_pred)):
        f.write(example_ids[i]+';;'+demo_pred[i]+'\n')
    f.close()



# svm try
def svm_trainNtest():
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



def dfs_trainNtest(dfs_y_train, dfs_y_test):
    model = DFS(in_dim = vocab_size, num_classes = 2, lambda1 = 0.04) #
    print(dfs_y_train)
    print(X_train)
    # prints
    model.fit(X_train, dfs_y_train, epochs = 10, batch_size = 100, validation_data = [X_test, dfs_y_test])
    print(model.accuracy(X_test, dfs_y_test))
    print(model.write_predictions("dfs_results.txt", X_test, dfs_y_test))



# **PREPROCESSING**
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
class LemmaTokenizer(object):
     def __init__(self):
         self.wnl = WordNetLemmatizer()
     def __call__(self, doc):
         return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]



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

# dataframe.text = dataframe.text.apply(lemmatize_text)



vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(), stop_words=stopset)
X = vectorizer.fit_transform(dataframe.text)
vocab_size = len(vectorizer.get_feature_names())
# X1 = vectorizer.fit_transform(df1.text)
# X2 = vectorizer.fit_transform(df2.text)
y = dataframe.classy
# y1 = df1.classy
# y2 = df2.classy
y_dfs = dataframe.classy
y_dfs = to_categorical(y_dfs)
print(y)


# Training on both provided data sets (X, y)... why not?
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
# nb_trainNtest()
print(y_train)

# X_train, X_test, dfs_y_train, dfs_y_test = train_test_split(X, y_dfs, test_size = 0.3)

#
# NEED to change demo_file string on demo day
demo_filename = 'training_data/data-1_train.csv'
#
nb_trainNtest()
nb_for_demo(demo_filename)
# dfs_trainNtest(dfs_y_train, dfs_y_test)
# svm_trainNtest()
