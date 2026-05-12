# Practical No: 4
# Title: Create a Linear Regression Model using Python/R to predict home prices
# using Boston Housing Dataset. The Boston Housing dataset contains information
# about various houses in Boston through different parameters. There are 506
# samples and 14 feature variables in this dataset. The objective is to predict
# the value of prices of the house using the given features.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load dataset (Boston removed -> use California Housing)
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()

data = pd.DataFrame(housing.data, columns=housing.feature_names)
data['PRICE'] = housing.target

# Step 2: Check data
print(data.head())
print(data.isnull().sum())

# Step 3: Split dependent and independent variables
x = data.drop('PRICE', axis=1)
y = data['PRICE']

# Step 4: Train-test split
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=0)

# Step 5: Train Linear Regression model
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(xtrain, ytrain)

# Step 6: Predictions
ytrain_pred = lm.predict(xtrain)
ytest_pred = lm.predict(xtest)

# Step 7: Evaluation
from sklearn.metrics import mean_squared_error, r2_score
mse_train = mean_squared_error(ytrain, ytrain_pred)
mse_test = mean_squared_error(ytest, ytest_pred)

r2_train = r2_score(ytrain, ytrain_pred)
r2_test = r2_score(ytest, ytest_pred)

print("Train MSE:", mse_train)
print("Train R2:", r2_train)
print("Test MSE:", mse_test)
print("Test R2:", r2_test)

# Step 8: Create DataFrames for comparison
df_train = pd.DataFrame({'Actual': ytrain, 'Predicted': ytrain_pred})
df_test = pd.DataFrame({'Actual': ytest, 'Predicted': ytest_pred})

# Step 9: Plot results
plt.scatter(ytrain, ytrain_pred, c='blue', marker='o', label='Training data')
plt.scatter(ytest, ytest_pred, c='lightgreen', marker='s', label='Test data')
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.title('True Value vs Predicted Value')
plt.legend(loc='upper left')
plt.show()
