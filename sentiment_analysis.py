#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn import naive_bayes
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

#read csv and label columns since default adds a space before some names
df = pd.read_csv('training_data/data-1_train.csv', names=['example_id', 'text', 'aspect_term', 'term_location', 'classy'])
#remove the first row since it's header duplication
df = df.drop([0], axis=0)


# In[16]:


stopset = set(stopwords.words('english'))
# stopset.add('[comma]')
y = df.classy
vectorizer = CountVectorizer(stop_words=stopset)
X = vectorizer.fit_transform(df.text)
print(X.toarray())

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

        
print(count)
print(len(pred))
print(count/len(pred))


# In[ ]:





# In[ ]:





# In[ ]:




