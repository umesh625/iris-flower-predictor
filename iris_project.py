#Libraries Importation
import pandas as pd               
import numpy as np                
import matplotlib.pyplot as plt  
import seaborn as sns             
import warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import load_iris                        
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler              
from sklearn.neighbors import KNeighborsClassifier             
from sklearn.linear_model import LogisticRegression            
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
import joblib  

#Data Collection and Exploration 
print("=" * 55)
print("Data Collection and Exploration")
print("=" * 55)

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print("\n First 5 rows of the dataset:")
print(df.head())

print("\n Dataset Shape (rows, columns):", df.shape)

print("\n Data Types:")
print(df.dtypes)

print("\n Summary Statistics:")
print(df.describe())

print("\n Missing Values per column:")
print(df.isnull().sum())

print("\n Species Distribution:")
print(df['species_name'].value_counts())

#Data Cleaning and Transformation 

print("\n" + "=" * 55)
print("Data Cleaning and Transformation")
print("=" * 55)

# Check for missing values
missing = df.isnull().sum().sum()
print(f"\n Total missing values: {missing} — No cleaning needed for Iris dataset!")

# Check for duplicates
duplicates = df.duplicated().sum()
print(f" Duplicate rows found: {duplicates}")
if duplicates > 0:
    df = df.drop_duplicates()
    print(f"   → Removed {duplicates} duplicate rows.")

# Outlier Detection using IQR method
print("\n Outlier Detection (IQR Method):")
feature_cols = iris.feature_names
for col in feature_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)].shape[0]
    print(f"   {col}: {outliers} outlier(s)")

# Exploratory Data Analysis (EDA)
print("\n" + "=" * 55)
print("Exploratory Data Analysis (EDA)")
print("=" * 55) 

#Distribution of each featur
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Feature Distributions by Species', fontsize=16, fontweight='bold')

for ax, col in zip(axes.flatten(), feature_cols):
    for species in df['species_name'].unique():
        subset = df[df['species_name'] == species][col]
        ax.hist(subset, alpha=0.6, label=species, bins=15)
    ax.set_title(col)
    ax.set_xlabel('Value (cm)')
    ax.set_ylabel('Frequency')
    ax.legend()

plt.tight_layout()
plt.savefig('eda_distributions.png', dpi=100)
plt.show()


#  Correlation Heatmap 
plt.figure(figsize=(8, 6))
corr_matrix = df[feature_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_correlation.png', dpi=100)
plt.show()


# Pairplot with kde
sns.pairplot(df[feature_cols + ['species_name']], hue='species_name', diag_kind='kde')
plt.suptitle('Pairplot of All Features', y=1.02, fontsize=14, fontweight='bold')
plt.savefig('kde_eda_pairplot.png', dpi=100)
plt.show()


#Pairplot with kde
sns.pairplot(df[feature_cols + ['species_name']], hue='species_name', diag_kind='hist')
plt.suptitle('Pairplot of All Features', y=1.02, fontsize=14, fontweight='bold')
plt.savefig('hist_eda_pairplot.png', dpi=100)
plt.show()


#  Boxplots 
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
fig.suptitle('Boxplots: Feature vs Species', fontsize=14, fontweight='bold')
for ax, col in zip(axes, feature_cols):
    df.boxplot(column=col, by='species_name', ax=ax)
    ax.set_xlabel('Species')
    ax.set_title(col)
plt.tight_layout()
plt.savefig('eda_boxplots.png', dpi=100)
plt.show()

print("\n Key EDA Findings:")
print("  Setosa is clearly separable from the other two species.")
print("  Petal length and petal width have the highest correlation (0.96).")
print("  Versicolor and Virginica overlap slightly — harder to separate.")
print("  Petal features are more discriminative than sepal features.")

#Feature Selecetion
print("\n" + "=" * 55)
print(" Feature Selection")
print("=" * 55)


X = df[feature_cols]
y = df['species']
print("\n Selected Features:", list(X.columns))
print(" Target Variable  : species (0=setosa, 1=versicolor, 2=virginica)")


#Model Development
print("\n" + "=" * 55)
print(" Model Development")
print("=" * 55)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n Training set : {X_train.shape[0]} samples")
print(f" Testing set  : {X_test.shape[0]} samples")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled  = scaler.transform(X_test)         
print("\n✅ Features scaled using StandardScaler")


# Model Training : K-Nearest Neighbors 
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
knn_preds = knn.predict(X_test_scaled)
knn_acc = accuracy_score(y_test, knn_preds)
print(f"\n KNN Accuracy: {knn_acc:.4f} ({knn_acc*100:.2f}%)")

# Model Training : Logistic Regression 
lr = LogisticRegression(max_iter=200, random_state=42)
lr.fit(X_train_scaled, y_train)
lr_preds = lr.predict(X_test_scaled)
lr_acc = accuracy_score(y_test, lr_preds)
print(f"Logistic Regression Accuracy: {lr_acc:.4f} ({lr_acc*100:.2f}%)")

# Model Evaluation and Tuning
print("\n" + "=" * 55)
print(" Model Evaluation and Hyperparameter Tuning")
print("=" * 55)

#Cross Validation
knn_cv = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='accuracy')
lr_cv  = cross_val_score(lr,  X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"\n KNN  5-Fold CV: {knn_cv.mean():.4f} ± {knn_cv.std():.4f}")
print(f"LR   5-Fold CV: {lr_cv.mean():.4f}  ± {lr_cv.std():.4f}")


#Hyper Parameter tuning for KNN using GreidSearchCV
print("\n Tuning KNN with GridSearchCV...")
param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)
best_k = grid_search.best_params_['n_neighbors']
best_knn = grid_search.best_estimator_
best_preds = best_knn.predict(X_test_scaled)
best_acc = accuracy_score(y_test, best_preds)
print(f"   Best K: {best_k}")
print(f"   Best KNN Accuracy after tuning: {best_acc:.4f} ({best_acc*100:.2f}%)")


# Confusion Matrix 
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, best_preds)
disp = ConfusionMatrixDisplay(cm, display_labels=['setosa', 'versicolor', 'virginica'])
disp.plot(ax=ax, cmap='Blues')
plt.title('Confusion Matrix - Best KNN Model', fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=100)
plt.show()
print(" Saved: confusion_matrix.png")

#Best Model Selection
best_model = best_knn
print(f"\n Best Model Selected: KNN (k={best_k}) with accuracy {best_acc*100:.2f}%")

# Save Model for Streamlit App
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("Save Model & Scaler")
print("=" * 55)

joblib.dump(best_model, 'iris_model.pkl')
joblib.dump(scaler,     'iris_scaler.pkl')
print(" Model saved  : iris_model.pkl")
print(" Scaler saved : iris_scaler.pkl")
print("\n  Run 'streamlit run app.py' to launch the web app.")