# Practical No. 5
# Name: Motkar Rushikesh Dinkar
#
# Title:
# 1. Implement logistic regression using Python/R to perform classification on
#    Social_Network_Ads.csv dataset.
# 2. Compute Confusion matrix to find TP, FP, TN, FN, Accuracy, Error rate,
#    Precision, Recall on the given dataset.

# Logistic Regression Model for Social Network Ads Prediction

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Load the dataset
df = pd.read_csv("Social_Network_ads (1).csv")

# Display the first few rows
print("Dataset Head:")
print(df.head())

# Encode the Gender column
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

# Define features and target
X = df[['User ID', 'Gender', 'Age', 'EstimatedSalary']]
y = df['Purchased']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Logistic Regression model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Make predictions on the test set
y_pred = logreg.predict(X_test)

# Test the prediction with new input data
print("\nTesting with new user data:")

# Example new user: User ID, Gender (0=Female, 1=Male), Age, EstimatedSalary
new_user_df = pd.DataFrame([[12345, 1, 30, 50000]],
                            columns=['User ID', 'Gender', 'Age', 'EstimatedSalary'])

# Make prediction
prediction = logreg.predict(new_user_df)
print(f"Prediction for new user: {'Purchased' if prediction[0] == 1 else 'Not Purchased'}")

# Also show probability
probabilities = logreg.predict_proba(new_user_df)
print(f"Probability of not purchasing: {probabilities[0][0]:.3f}")
print(f"Probability of purchasing: {probabilities[0][1]:.3f}")

# Evaluate the performance of the logistic regression model
print("\nModel Evaluation:")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Extract TP, FP, TN, FN
tn, fp, fn, tp = cm.ravel()
print(f"\nTP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")

# Error Rate
error_rate = 1 - accuracy
print(f"Error Rate: {error_rate:.4f}")

# Precision and Recall
precision = tp / (tp + fp)
recall = tp / (tp + fn)
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
