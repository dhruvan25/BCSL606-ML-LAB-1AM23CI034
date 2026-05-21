# Support Vector Machine (SVM) Classification

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
dataset = pd.read_csv("Social_Network_Ads.csv")

# Input and output split
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Train-test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=0
)

# Feature scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Import SVM model
from sklearn.svm import SVC

# Create model
classifier = SVC(
    kernel='rbf',
    random_state=0
)

# Train model
classifier.fit(X_train, y_train)

# Predict test results
y_pred = classifier.predict(X_test)

# Print predictions
print("Predictions:")
print(y_pred)

# Accuracy
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

# Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Sample prediction
sample = [[30, 87000]]

sample_scaled = sc.transform(sample)

prediction = classifier.predict(sample_scaled)

print("\nPrediction for sample [30, 87000]:")
print(prediction)