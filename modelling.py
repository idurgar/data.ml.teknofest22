import pandas as pd
import re
import numpy as np
import nltk
import pickle
import numpy as np
import text_preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# import models
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
import fasttext

# read data
df = pd.read_csv(data_path + "kanunum-nlp-doc-analysis-dataset.csv")

# preprocessing
text_preprocessing.preprocessing(df)

# BASIC MODELLING

## text to numerical array
vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7)
X = vectorizer.fit_transform(df.cleaned_text.values).toarray()

# splitting
X_train, X_test, y_train, y_test = train_test_split(X, df.kategori, test_size=0.2, random_state=0)

# CREATE BASE MODELS

## 1. TRAINING BASIC LinearSVC WITHOUT HYPERPARAMETER TUNING and SHOW RESULTS
classifier = LinearSVC()
classifier.fit(X_train, y_train) 
y_pred = classifier.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))


# 2. FASTTEXT MODELLING

# convert "kategori column" to integers
df_temp = df.copy()
df_temp['labels'] = pd.factorize(df_temp.kategori)[0] 
train, test = train_test_split(df_temp[["kategori", "cleaned_text", "labels"]], test_size=0.2, random_state=42)
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

# Preparing train data to fasttext format
train["label_format"]=0
for i in range(len(train)):
    train.label_format[i]="__label__"+str(train.kategori[i])+" "+str(train.cleaned_text[i])
    
# Preparing test data to fasttext format
test["label_format"]=0
for i in range(len(test)):
    test.label_format[i]="__label__"+str(test.kategori[i])+" "+str(test.cleaned_text[i])  
 
train.label_format.to_csv('fasttext_train.txt',index=None,header=None)
test.label_format.to_csv('fasttext_test.txt',index=None,header=None) 

# training fasttext    
model = fasttext.train_supervised('fasttext_train.txt',epoch=50,lr=0.05,label_prefix='__label__',dim=300)
# testing
model.test('fasttext_test.txt')    