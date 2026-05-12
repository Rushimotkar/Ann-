# Practical No. 9
# Title: Data Visualization II
#
# 1) Use the inbuilt dataset 'titanic' as used in the above problem.
#    Plot a box plot for distribution of age with respect to each gender
#    along with information about whether they survived or not.

import seaborn as sns
import matplotlib.pyplot as plt

# Load the Titanic dataset
titanic = sns.load_dataset('titanic')

print("Dataset Shape:", titanic.shape)
print("\nFirst 5 rows:")
print(titanic.head())

# Box plot: Age distribution by Gender and Survival status
plt.figure(figsize=(9, 6))
sns.boxplot(x='sex', y='age', hue='survived', data=titanic, palette='Set2')
plt.title("Age Distribution by Gender and Survival Status\n(0 = Not Survived, 1 = Survived)")
plt.xlabel("Gender")
plt.ylabel("Age")
plt.legend(title='Survived', labels=['Not Survived', 'Survived'])
plt.show()
