# Random Forest Classifier

import pandas as pd
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

# Random Forest model
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(
    n_estimators=100,
    criterion='entropy',
    random_state=0
)

# Train model
classifier.fit(X_train, y_train)

# Prediction
y_pred = classifier.predict(X_test)

print("Predictions:")
print(y_pred)

# Accuracy
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)