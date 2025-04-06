# Heart Disease Detection using Machine Learning Algorithms

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Suppress warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('.\\dataset\\dataset.csv')

# Drop unnecessary columns
if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)

# Convert 'num' column to binary target (0 = No disease, 1 = Disease)
df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)

# Handle missing values by imputing with column means
df.fillna(df.mean(numeric_only=True), inplace=True)

# Show dataset summary
print("Dataset Description:\n")
print(df.describe())

# --- Visualization Section ---

# Correlation heatmap (numeric features only)
plt.figure(figsize=(18, 18))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap='rocket', square=True)
plt.title("Correlation Heatmap (Numeric Features Only)")
plt.show()

# Feature distributions
df.hist(figsize=(20, 20), bins=30, edgecolor='black')
plt.suptitle("Feature Distributions", fontsize=20)
plt.show()

# Count plot of heart disease presence
sns.set_style('whitegrid')
sns.countplot(x='target', data=df, palette='flare')
plt.title("Heart Disease Distribution (0: No, 1: Yes)")
plt.xlabel("Target")
plt.ylabel("Count")
plt.show()

# --- Preprocessing Section ---

# One-hot encode categorical features (if any)
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Feature scaling
columns_to_scale = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
scaler = StandardScaler()
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

# Split into features and target
X = df.drop('target', axis=1)
y = df['target']

# --- KNN Analysis ---

knn_scores = []

# Try K values from 1 to 10
for k in range(1, 11):
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X, y, cv=10)
    knn_scores.append(score.mean())

# Plot KNN scores
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), knn_scores, marker='o', color='red')
for i in range(1, 11):
    plt.text(i, knn_scores[i-1], f"{knn_scores[i-1]:.2f}", ha='center', va='bottom')
plt.title("KNN Accuracy for Different K Values")
plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Cross-Validation Accuracy")
plt.grid(True)
plt.xticks(range(1, 11))
plt.show()

# Best K result
best_k = knn_scores.index(max(knn_scores)) + 1
print(f"\nBest K Value: {best_k} with Accuracy: {max(knn_scores):.3f}")

# --- Random Forest Classifier ---

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_score = cross_val_score(rf_model, X, y, cv=10)

print(f"\nRandom Forest Mean Accuracy: {rf_score.mean():.3f}")
