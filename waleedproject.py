

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
# Load Dataset
df = pd.read_csv('Cleaned_Insurance_Data_Final.csv')
# Data Preprocessing
le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])
df['smoker'] = le.fit_transform(df['smoker'])
df['region'] = le.fit_transform(df['region'])
# Feature Selection
X = df.drop(columns=['expenses'])  # Features
y = df['expenses'] > df['expenses'].median()  # Binary Classification
# Splitting Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Hyperparameter tuning for KNN
param_grid = {'n_neighbors': range(1, 21), 'weights': ['uniform', 'distance']}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_knn = grid_search.best_estimator_
# Model Training - Naïve Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)
nb_accuracy = accuracy_score(y_test, nb_pred)
print("Naïve Bayes Accuracy:", nb_accuracy)
# Model Training - Optimized KNN
knn_model = best_knn
knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_pred)
print("Optimized KNN Accuracy:", knn_accuracy)
# Evaluation
print("Naïve Bayes Classification Report:")
print(classification_report(y_test, nb_pred))
print("Optimized KNN Classification Report:")
print(classification_report(y_test, knn_pred))
# Confusion Matrices
plt.figure(figsize=(10,5))
sns.heatmap(confusion_matrix(y_test, nb_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Naïve Bayes Confusion Matrix')
plt.show()
plt.figure(figsize=(10,5))
sns.heatmap(confusion_matrix(y_test, knn_pred), annot=True, fmt='d', cmap='Reds')
plt.title('Optimized KNN Confusion Matrix')
plt.show()
# Save Models
joblib.dump(nb_model, 'naive_bayes_model.pkl')
joblib.dump(knn_model, 'knn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')