# PRACTICAL NO - 03
# Title: Descriptive Statistics - Measures of Central Tendency and variability
# perform the following operations on any open source dataset.
#
# Write a Python program to display some basic statistical details like percentile,
# mean, standard deviation, etc. of the species of "Iris-setosa", "Iris-versicolor"
# and "Iris-versicolor" of iris.csv dataset.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# Load the dataset
df = pd.read_csv('Iris.csv')

# Split the dataset into features and labels
X = df.drop('Species', axis=1)
y = df['Species']

print("Features (X):")
print(X)

print("\nLabels (y):")
print(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Gaussian Naive Bayes classifier
classifier = GaussianNB()

# Train the classifier
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Confusion Matrix
confusion_mat = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(confusion_mat)

# For binary-style metric calculation using confusion matrix values
# Note: For multiclass, we use sklearn's built-in functions below
tn, fp, fn, tp = confusion_mat[0, 0], confusion_mat[0, 1], confusion_mat[1, 0], confusion_mat[1, 1]

# Compute evaluation metrics (manual)
accuracy = (tp + tn) / (tp + tn + fp + fn)
error_rate = 1 - accuracy
precision = tp / (tp + fp)
recall = tp / (tp + fn)

# Print evaluation metrics
print("\nAccuracy:", accuracy)
print("Error Rate:", error_rate)
print("Precision:", precision)
print("Recall:", recall)
