# -*- coding: utf-8 -*-
"""

Ma412 main

ABDELMALEK Enzo
ROBINEAU Eliott 
Sialelli Janelle

This file contains every test presented in the report of the project.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import fasttext
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import Ma412_lib as l
import Ma412_class as c

# %% Loading the data

# If the loading doesn't work, add the parameter engine = "pyarrow" in the pd.read_paraquet function
df = pd.read_parquet("Data/train-00000-of-00001-b21313e511aa601a.parquet")
df = df.dropna(subset=["title", "abstract"]).reset_index(drop=True) # Removing the raws where there is no abstract or no title
dv = pd.read_parquet("Data/val-00000-of-00001-66ce8665444026dc.parquet")
dv = dv.dropna(subset=["title", "abstract"]).reset_index(drop=True) # Removing the raws where there is no abstract or no title

# %% Preprocessing of the data 

# Creation of an empty dictionary that will store the labels and their occurence in the database
labels = {}

# We retrieve every labels from our databse 
lab = []
for i in range(len(df["verified_uat_labels"])):
    for j in df["verified_uat_labels"][i]:
        lab.append(j)

# Creation of an encoder to encod the labels
encod = LabelEncoder()
# Enconding the labels
labels_unique = encod.fit_transform(lab)

# Determining the number of unique labels
for i in range(len(labels_unique)):
    word = lab[i]
    if word not in labels.keys():
        labels[word] = 0
nb_unique_labels = max(labels_unique) + 1

# Determining the occurence of each label in the database
for i in lab:
    labels[i] += 1

# Final labels dictionnary
labels = dict(sorted(labels.items(), key=lambda x: x[1], reverse=True))
keys = list(labels.keys())

# Abstracts and titles recovery
abstracts = list(df["abstract"])
titles = list(df["title"])

#%% Histogram of the ten most labels used in the database

l.Display_labels(labels,10)

# %% Examining the presence of the labels in the abstracts

y_pred = np.zeros((len(abstracts), nb_unique_labels))
ab_count = 0
for abstract in abstracts:
    lab_count = 0
    for label in labels.keys():
        if label in abstract:
            y_pred[ab_count, lab_count] = 1
        lab_count += 1
    ab_count += 1

# %% Highlightning of the first problem

plt.imshow(y_pred)

# We get the prediction associated to the first abstract
lab_abstract_1 = y_pred[1, :]

# We get the positions associated to the detected labels
indices = np.argwhere(lab_abstract_1 != 0).reshape(-1,)
labels_identifies = [keys[i] for i in indices]  # Labels recovery
abstract_1 = abstracts[0]  # First abstract
pourcentage_occurence_io = (np.sum(y_pred[:, 992])/np.shape(y_pred)[0])*100

print("Labels found in the first abstract: " +
      str(labels_identifies) + "\n")  # Highlightning of the first problem
print("Occurrence of the label 'io' in the abstracts: " +
      str(np.round(pourcentage_occurence_io, 2)) + "%")

# %% Highlightning of the second and third problems

# We get the prediction associated to the second abstract
lab_abstract_3 = y_pred[2, :]

# We get the positions associated to the detected labels
indices = np.argwhere(lab_abstract_3 != 0).reshape(-1,)
labels_identifies = [keys[i] for i in indices]  # Labels recovery
# We get the labels that have been provided by the authors
labels_given_3 = list(df["verified_uat_labels"])[2]

print("Labels intially given by the authors of the abstract: \n" +
      str(labels_given_3) + "\n")
# Second problem emphasized : problem of upper case
print("Presence of the word 'europa' in the abstract: " +
      str('europa' in abstracts[2]) + '\n')
# Second problem emphasized : problem of upper case
print("Presence of the word 'Europa' in the abstract: " +
      str('Europa' in abstracts[2]) + '\n')
print("Labels found in the third abstract: \n" + str(labels_identifies) + "\n")
print("Labels found in the third abstract compared with the ones originally given by the authors: " +
      str([label in list(df["verified_uat_labels"])[2] for label in labels_identifies]))  # Third problem emphasized : Non presence of labels in the abstract

#%% First model 

# Number of articles we consider for the training

N_train = 3000

# Restriction of the abstracts and titles to the number of articles considered
abstracts = abstracts[:N_train]
titles = titles[:N_train]

# Normalization of the labels, abstracts and titles for the algorithm
label_norm = list(df["verified_uat_labels"])[:N_train]
abstract_norm = [l.text_to_id(abstract) for abstract in abstracts]
title_norm = [l.text_to_id(title) for title in titles]

# Combination of the titles and the abstracts to make the corpus
corpus_norm = [abst + " " + titl for abst, titl in zip(abstract_norm, title_norm)]

# Creation of and training of the corpus vectorizer using TF-IDF with a max feature of 1000
corpus_vectorizer = TfidfVectorizer(max_features = 1000)
corpus_vectorizer.fit(corpus_norm)

# Transformating the TF-IDF vectorizer 
X = corpus_vectorizer.transform(corpus_norm).toarray()

# Creation of a binary vector for each scientific paper and each label
y = np.zeros((len(label_norm), len(labels)))  

for idx, labels_in_example in enumerate(label_norm):
    for label in labels_in_example:
        y[idx, labels[label]] = 1  # Set the labels associated with this paper

# Splitting the data into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Parameters for the class LogisticRegression
nb_iter = 100000
alpha = 1e-3

logistic_regression = c.LogisticRegression(nb_iter, alpha, X_train)

#logistic_regression.train(X_train, y_train)

# Save the logistic regression parameters after trainin the model
#np.savez('logistic_regression_params.npz', w=logistic_regression.w, bias=logistic_regression.bias, 
#         nb_iter = logistic_regression.nb_iter, losses = logistic_regression.losses, alpha = logistic_regression.alpha)

# Load the trained logistic regression parameters 
data = np.load('logistic_regression_params.npz')
logistic_regression.w = data['w']
logistic_regression.bias = data['bias']

Z = logistic_regression.compute_Z(X_test)
y_pred2 = logistic_regression._sigmoid(Z)

s1 = np.sum(y_pred2[:,7])
s2 = np.sum(y_train[:,7])

print("\nThe sum of the 7th column for the prediction of the test data: " + str(np.round(s1,2)))
print("Half the sum of the 7th column of the train data: " + str(s2/2))

y_pred = logistic_regression.predict(X_test, threshold = 0.035)

# Computation of the f1-micro score
f1_micro = f1_score(y_test, y_pred, average='micro')
print("\nThe f1-micro score for the model is : f1-micro = " + str(np.round(f1_micro*100,2)) + "%")

# %% Second model

N_train = df.shape[0]

# Output file
output_file = "fasttext_data.txt"

l.prepare_fasttext_data(df["title"], abstracts, df["verified_uat_labels"], output_file, N_train)

input_file = "fasttext_data.txt"
output_file = "fasttext_preprocessed.txt"

# Creation of the text for the supervised training
l.fasttext_preprocessed(output_file,input_file)

# Supervised training
model = fasttext.train_supervised(input="fasttext_preprocessed.txt", lr=0.1, epoch=25, wordNgrams=2, bucket=200000, dim=300, loss='ova')
model2 = fasttext.train_supervised(input="fasttext_preprocessed.txt", lr=0.01, epoch=75, wordNgrams=2, bucket=200000, dim=300, loss='ova')
model3 = fasttext.train_supervised(input="fasttext_preprocessed.txt", lr=0.001, epoch=50, wordNgrams=2, bucket=200000, dim=300, loss='ova')
model4 = fasttext.train_supervised(input="fasttext_preprocessed.txt", lr=0.001, epoch=25, wordNgrams=2, bucket=200000, dim=300, loss='ova')

N = dv.shape[0]

# Output file validation
output_file = "validation.txt"

abstracts = list(dv["abstract"])
abstracts = [x for x in abstracts if x is not None]

l.prepare_fasttext_data(dv["title"], abstracts, dv["verified_uat_labels"], output_file,N)

input_file = "validation.txt"
output_file = "validation_set.txt"

# Creation of the text file to test the model
l.fasttext_preprocessed(output_file,input_file)

# Testing the model
N_sample, Precision_at_one, Recall_at_one = model.test("validation_set.txt", k=-1)
N_sample, Precision_at_one2, Recall_at_one2 = model2.test("validation_set.txt", k=-1)
N_sample, Precision_at_one3, Recall_at_one3 = model3.test("validation_set.txt", k=-1)
N_sample, Precision_at_one4, Recall_at_one4 = model4.test("validation_set.txt", k=-1)
print("The precision at one for the first model is given by:" + str(Precision_at_one))
print("The recall at one for the first model is given by:" + str(Recall_at_one) + "\n")

print("The precision at one for the second model is given by:" + str(Precision_at_one2))
print("The recall at one for the second model is given by:" + str(Recall_at_one2) + "\n")

print("The precision at one for the third model is given by:" + str(Precision_at_one3))
print("The recall at one for the third model is given by:" + str(Recall_at_one3) + "\n")

print("The precision at one for the fourth model is given by:" + str(Precision_at_one4))
print("The recall at one for the fourth model is given by:" + str(Recall_at_one4) + "\n")
