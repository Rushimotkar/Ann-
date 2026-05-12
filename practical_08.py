# Practical No. 8
# Name: Motkar Rushikesh Dinkar
#
# Title: Data Visualization I
# 1. Use the inbuilt dataset 'titanic'. The dataset contains 891 rows and contains
#    information about the passengers who boarded the unfortunate Titanic ship.
#    Use the Seaborn library to see if we can find any patterns in the data.
# 2. Write a code to check how the price of the ticket (column name: 'fare') for
#    each passenger is distributed by plotting a histogram.

import seaborn as sns
import matplotlib.pyplot as plt

# Load the Titanic dataset
titanic = sns.load_dataset('titanic')

print("Dataset Shape:", titanic.shape)
print("\nFirst 5 rows:")
print(titanic.head())

# 1. Survival count
sns.countplot(x='survived', data=titanic)
plt.title("Survival Count (0 = Not Survived, 1 = Survived)")
plt.show()

# 2. Survival based on class
sns.countplot(x='pclass', hue='survived', data=titanic)
plt.title("Survival based on Passenger Class")
plt.show()

# 3. Survival based on gender
sns.countplot(x='sex', hue='survived', data=titanic)
plt.title("Survival based on Gender")
plt.show()

# 4. Age distribution
sns.histplot(titanic['age'].dropna(), kde=True)
plt.title("Age Distribution")
plt.show()

# 5. Fare distribution histogram
plt.figure(figsize=(8, 5))
sns.histplot(titanic['fare'], bins=30, kde=True, color='blue')
plt.title("Fare Distribution of Titanic Passengers")
plt.xlabel("Fare")
plt.ylabel("Count")
plt.show()
