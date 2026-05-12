# PRACTICAL NO - 02
# AIM: Create an "Academic performance" dataset of students and perform the
#      following operations using Python.

import pandas as pd
import numpy as np

data = {
    'ROLL_NO': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Stud_Name': ['SAKSHI', 'GAURI', 'ANJALI', 'MANSI', 'SHREYA',
                  'Samantha', 'ADITI', 'PALLAVI', 'RIYA', 'BHUMIKA'],
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male',
               'Female', 'Male', 'Female', 'Male', 'Female'],
    'Age': [20, 19, 20, 19, 21, 20, 21, 19, 20, 20],
    'Class': ['TE', 'BE', 'TE', 'SE', 'TE', 'FE', 'SE', 'TE', 'SE', 'FE'],
    'Major': ['Science', 'Arts', 'Engineering', 'Business', 'Science',
              'Arts', 'Engineering', 'Business', 'Science', 'Arts'],
    'GPA': [3.2, 3.5, 3.8, 2.9, 3.6, 3.1, 3.7, 3.9, 3.5, 3.3],
    'Insem_Score': [29, 25, 27, 22, 18, 20, 22, 25, 23, 27],
    'Attendance': [90, 87, 85, 95, 73, 60, 100, 95, 85, 80]
}

df = pd.DataFrame(data)

# Check for missing values
print("Missing Values:")
print(df.isnull().sum())

# Check for inconsistencies in categorical variables
print("\nGender Value Counts:")
print(df['Gender'].value_counts())

# Check for inconsistencies in numerical variables
print("\nNumerical Variables Description:")
print(df[['Age', 'GPA', 'Insem_Score', 'Attendance']].describe())

import matplotlib.pyplot as plt

# Create boxplot for numerical variables
df[['Age', 'GPA', 'Insem_Score', 'Attendance']].boxplot()
plt.title('Boxplot of Numerical Variables')
plt.show()

# Create scatter plot for GPA and Insem Score
plt.scatter(df['GPA'], df['Insem_Score'])
plt.xlabel('GPA')
plt.ylabel('Insem_Score')
plt.title('GPA vs Insem Score')
plt.show()

# Apply logarithmic transformation to 'Insem Score' variable
df['Log Insem_Score'] = np.log(df['Insem_Score'])
print("\nDataFrame with Log Insem_Score:")
print(df[['Stud_Name', 'Insem_Score', 'Log Insem_Score']])
