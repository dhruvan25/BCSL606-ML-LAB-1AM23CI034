import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load Dataset
# Make sure Salary_Data.csv is in the same folder

dataset = pd.read_csv("Salary_Data.csv")

# Input and Output Split
X = dataset[["YearsExperience"]]
y = dataset["Salary"]

# Split Dataset into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=0
)

# Create Model
model = LinearRegression()

# Train Model
model.fit(X_train, y_train)

# Predict Values
y_pred = model.predict(X_test)

# Print Predictions
print("Predicted Salary Values:")
print(y_pred)

# Print Coefficients
print("\nSlope (Coefficient):", model.coef_)
print("Intercept:", model.intercept_)

# Visualize Training Set Results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, model.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()