"""
Data Visualization Practical - Iris Flowers Dataset
Vishwabharati Academy's College of Engineering, Ahilyanagar
Practical No: 10

Tasks:
1. Download the Iris Flowers dataset
2. List down the features and their types
3. Create a histogram for each feature
4. Create a box plot for each feature in dataset
5. Compare distributions & identify outliers
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# ============================================================
# TASK 1: Download / Load the Iris Flowers Dataset
# ============================================================
print("=" * 60)
print("TASK 1: Loading the Iris Flowers Dataset")
print("=" * 60)

# Load the Iris dataset from sklearn
iris = load_iris()

# Create a DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = [iris.target_names[i] for i in iris.target]

print("\nDataset loaded successfully!")
print(f"Shape of dataset: {df.shape}")
print(f"Total samples: {df.shape[0]}")
print(f"Total features (including target): {df.shape[1]}")

# ============================================================
# TASK 2: List down the features and their types
# ============================================================
print("\n" + "=" * 60)
print("TASK 2: Features and Their Data Types")
print("=" * 60)

print("\n--- Data Types of Each Feature ---")
print(df.dtypes)

print("\n--- Feature Descriptions ---")
feature_info = {
    'sepal length (cm)': 'Continuous (Numeric) - Length of the sepal in centimeters',
    'sepal width (cm)': 'Continuous (Numeric) - Width of the sepal in centimeters',
    'petal length (cm)': 'Continuous (Numeric) - Length of the petal in centimeters',
    'petal width (cm)': 'Continuous (Numeric) - Width of the petal in centimeters',
    'species': 'Categorical (Nominal) - Species of Iris flower (setosa, versicolor, virginica)'
}

for feature, description in feature_info.items():
    print(f"\n  {feature}:")
    print(f"    Type: {description}")

print("\n--- Basic Statistical Summary ---")
print(df.describe())

print("\n--- Species Distribution ---")
print(df['species'].value_counts())

# ============================================================
# TASK 3: Create a histogram for each feature
# ============================================================
print("\n" + "=" * 60)
print("TASK 3: Creating Histograms for Each Feature")
print("=" * 60)

# Get numeric features only
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Histograms of Iris Dataset Features', fontsize=16, fontweight='bold')

for idx, feature in enumerate(numeric_features):
    ax = axes[idx // 2, idx % 2]

    # Plot histogram for each species with different colors
    for species in df['species'].unique():
        subset = df[df['species'] == species]
        ax.hist(subset[feature], bins=15, alpha=0.6, label=species, edgecolor='black')

    ax.set_xlabel(feature, fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'Distribution of {feature}', fontsize=12)
    ax.legend(title='Species')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('iris_histograms.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nHistograms saved as 'iris_histograms.png'")

# ============================================================
# TASK 4: Create a box plot for each feature
# ============================================================
print("\n" + "=" * 60)
print("TASK 4: Creating Box Plots for Each Feature")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Box Plots of Iris Dataset Features', fontsize=16, fontweight='bold')

for idx, feature in enumerate(numeric_features):
    ax = axes[idx // 2, idx % 2]

    # Create box plot grouped by species
    sns.boxplot(data=df, x='species', y=feature, ax=ax, palette='Set2')
    ax.set_title(f'Box Plot of {feature}', fontsize=12)
    ax.set_xlabel('Species', fontsize=11)
    ax.set_ylabel(feature, fontsize=11)

    # Add mean markers
    means = df.groupby('species')[feature].mean()
    for i, mean_val in enumerate(means):
        ax.scatter(i, mean_val, color='red', marker='D', s=50, zorder=5, label='Mean' if i == 0 else "")

    ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig('iris_boxplots.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nBox plots saved as 'iris_boxplots.png'")

# ============================================================
# TASK 5: Compare distributions & identify outliers
# ============================================================
print("\n" + "=" * 60)
print("TASK 5: Distribution Comparison & Outlier Detection")
print("=" * 60)

# Function to detect outliers using IQR method
def detect_outliers_iqr(data, feature):
    Q1 = data[feature].quantile(0.25)
    Q3 = data[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[feature] < lower_bound) | (data[feature] > upper_bound)]
    return outliers, lower_bound, upper_bound

print("\n--- Outlier Detection using IQR Method ---")
for feature in numeric_features:
    print(f"\n  Feature: {feature}")
    outliers, lb, ub = detect_outliers_iqr(df, feature)
    print(f"    Lower Bound: {lb:.3f}, Upper Bound: {ub:.3f}")
    if len(outliers) > 0:
        print(f"    Outliers Found: {len(outliers)} rows")
        for _, row in outliers.iterrows():
            print(f"      -> {row['species']}: {row[feature]:.2f} cm")
    else:
        print(f"    Outliers Found: None")

# Overall comparison
print("\n--- Distribution Comparison Summary ---")
print("\n  Mean values by species:")
print(df.groupby('species')[numeric_features].mean().round(3))

print("\n  Standard deviation by species:")
print(df.groupby('species')[numeric_features].std().round(3))

print("\n  Range (Min-Max) by species:")
for species in df['species'].unique():
    subset = df[df['species'] == species]
    print(f"\n    {species}:")
    for feature in numeric_features:
        print(f"      {feature}: {subset[feature].min():.2f} - {subset[feature].max():.2f}")

# Combined violin plot for better distribution comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Distribution Comparison - Violin Plots', fontsize=16, fontweight='bold')

for idx, feature in enumerate(numeric_features):
    ax = axes[idx // 2, idx % 2]
    sns.violinplot(data=df, x='species', y=feature, ax=ax, palette='muted', inner='box')
    ax.set_title(f'Distribution of {feature}', fontsize=12)
    ax.set_xlabel('Species', fontsize=11)
    ax.set_ylabel(feature, fontsize=11)

plt.tight_layout()
plt.savefig('iris_violinplots.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nViolin plots saved as 'iris_violinplots.png'")

# Correlation heatmap
print("\n--- Feature Correlation ---")
plt.figure(figsize=(8, 6))
correlation_matrix = df[numeric_features].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, fmt='.3f')
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('iris_correlation.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nCorrelation heatmap saved as 'iris_correlation.png'")

print("\n" + "=" * 60)
print("All tasks completed successfully!")
print("Generated files:")
print("  - iris_histograms.png")
print("  - iris_boxplots.png")
print("  - iris_violinplots.png")
print("  - iris_correlation.png")
print("=" * 60)
