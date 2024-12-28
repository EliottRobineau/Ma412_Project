# -*- coding: utf-8 -*-
"""

Ma412 class

ABDELMALEK Enzo
ROBINEAU Eliott 
Sialelli Janelle

This file contains every class presented in the report of the project.

"""

import numpy as np

#%% 

class LogisticRegression:

    def __init__(self, nb_iter, alpha, X):
        self.n_samples, self.n_features = X.shape
        self.nb_iter = nb_iter
        self.alpha = alpha  # alpha is the learning rate
        self.w = None
        self.bias = None
        self.losses = []

    # Compute the linear relation between X and Z
    def compute_Z(self, X):
        Z = np.dot(X, self.w) + self.bias
        return Z

    # Sigmoid activation function
    def _sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    # Compute the loss
    def compute_loss(self, y_true, y_pred):
        y_1 = y_true * np.log(y_pred)
        y_2 = (1 - y_true) * np.log(1 - y_pred)
        # L = (-1/2)*(y_1 + y_2)
        L = - np.mean(np.sum(y_1 + y_2, axis=1))
        return L

    def train(self, X, y):
        
        # Initialize w and b randomly if they are None
        if self.w is None and self.bias is None:
            self.w = np.zeros((self.n_features, y.shape[1]))  # One weight vector per label
            self.bias = np.zeros(y.shape[1])  # One bias term per label
        
        # Gradient descent
        for i in range(self.nb_iter):

            Z = self.compute_Z(X)
            a = self._sigmoid(Z)  # the predictions for all labels
            # Compute the loss to compare the predicted value with the real value
            loss = self.compute_loss(y, a)
            self.losses.append(loss)
            # Compute the gradients
            dz = a - y
            dW = (1 / self.n_samples) * np.dot(X.T, dz)
            db = (1 / self.n_samples) * np.sum(dz, axis=0)

            # Update the parameters
            self.w = self.w - self.alpha * dW
            self.bias = self.bias - self.alpha * db
            print("Training of the LogisticRegression iteration: " + str(i))
            print("Loss: " + str(loss))

    def predict(self, X, threshold=0.5):
        # Compute predictions for all classes
        Z = self.compute_Z(X)
        y_pred = self._sigmoid(Z)

        # Apply threshold to obtain binary predictions (multi-label)
        y_pred_label = (y_pred > threshold).astype(int)

        return y_pred_label
