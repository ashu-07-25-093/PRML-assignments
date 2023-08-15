import numpy as np
import re
import nltk
import pandas as pd

from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
import pickle

nltk.download('stopwords')      # downloading all stopwords to check this words and remove from email
nltk.download('punkt')

# reading the csv file
df = pd.read_csv("completeSpamAssassin.csv", encoding="ISO-8859-1")
df.drop(['Unnamed: 0'], axis=1, inplace=True)

# this function will first split the sentence into words and then remove the stopwords
def removeStopWords(email):

    # creating the list of stopwords
    stop_words = set(stopwords.words('english') + ['u', 'ü', 'ur', '4', '2', 'im', 'dont', 'doin', 'ure'])

    # tokenize the whole email into tokens
    tokens = word_tokenize(email)

    cleaned_tokens = []
    for word in tokens:
        if word not in stop_words:
            cleaned_tokens.append(word)

    return " ".join(cleaned_tokens)

# this function will replace website by word 'webaddress', email_id by word 'emailaddress', dollar symbol by word 'dollar',
# 10 digit phonenumber by word 'phonenumber', and any number by word 'number', and removing the extra spaces in between or
# from the beginning and at the end
def preprocessing(email_contents):
    email_contents = email_contents.lower()

    email_contents = re.sub('^.+@[^\.].*\.[a-z]{2,}$', 'emailaddress', email_contents)
    email_contents = re.sub('^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$', 'webaddress', email_contents)
    email_contents = re.sub('[£]+|[$]+', 'dollar', email_contents)
    email_contents = re.sub('^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$', 'phonenumber', email_contents)
    email_contents = re.sub('\d+(\.\d+)?', 'number', email_contents)

    # remove punctuation
    email_contents = re.sub('[^\w\d\s]', ' ', email_contents)
    # replace more than one space to single space
    email_contents = re.sub('\s+', ' ', email_contents)
    # remove extra spaces from beginning and end
    email_contents = re.sub('^\s+|\s+?$', '', email_contents)

    return email_contents

# calling the preprocessing and removeStopword functions
for idx in df.index:
    df['Body'][idx] = preprocessing(df['Body'][idx])
    df['Body'][idx] = removeStopWords(df['Body'][idx])

# separating the email in X and labels in Y
X = df['Body']
y_act = df['Label']

# splitting the data into train and validation, by 75% training and 25% validaiton
X_train, X_val, Y_train, Y_val = train_test_split(X, y_act, random_state=0 ,test_size=0.25,shuffle=True)

# countvectorizer, extracts the feature, how many times the word appear(frequency)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)

# Term frequency inverse document frequency, this will assign the less value to a word that is appears the most times and
# not much important
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

X_count_val = count_vect.transform(X_val)
X_tfidf_val = tfidf_transformer.transform(X_count_val)

# Support vector machine implementation
# by radial basis kernel
svc_rbf = svm.SVC(C = 3.0, kernel='rbf', random_state = 42)
svc_rbf.fit(X_train_tfidf, Y_train)

pred_val_rbf = svc_rbf.predict(X_tfidf_val)
myScore = accuracy_score(Y_val, pred_val_rbf)
print("Validation accuracy by radial basis kernel : ", myScore)
myModel = svc_rbf

#print(X_tfidf_val.shape)
# by polynomial kernel with degree 2
svc_poly2 = svm.SVC(kernel='poly', degree = 2, random_state = 42)
svc_poly2.fit(X_train_tfidf, Y_train)

pred_val_poly2 = svc_poly2.predict(X_tfidf_val)
if(accuracy_score(Y_val, pred_val_poly2) > myScore):
    myScore = accuracy_score(Y_val, pred_val_poly2)
    myModel = svc_poly2

print("Validation accuracy by Polynomial kernel with degree 2 : ", accuracy_score(Y_val, pred_val_poly2))

# by polynomial kernel with degree 2
svc_poly3 = svm.SVC(kernel='poly', degree = 3, random_state = 42)
svc_poly3.fit(X_train_tfidf, Y_train)

pred_val_poly3 = svc_poly3.predict(X_tfidf_val)
if(accuracy_score(Y_val, pred_val_poly3) > myScore):
    myScore = accuracy_score(Y_val, pred_val_poly3)
    myModel = svc_poly3

print("Validation accuracy by Polynomial kernel with degree 3 : ", accuracy_score(Y_val, pred_val_poly3))

# saving the model for prediction
filename_model = 'model.pkl'
pickle.dump(myModel, open(filename_model, 'wb'))  # store best model's  object to disk

# saving the count vectorizer for prediction
filename_count_vect = 'count_vectorizer.pkl'
pickle.dump(count_vect, open(filename_count_vect, 'wb'))

# saving the tfidf for prediction
filename_tfidf = 'tfidf_transformer.pkl'
pickle.dump(tfidf_transformer, open(filename_tfidf, 'wb'))  # store feature extractor to disk