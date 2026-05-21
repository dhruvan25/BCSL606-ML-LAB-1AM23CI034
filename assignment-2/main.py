# Multiple Linear Regression

# Import libraries
import pandas as pd

# Load dataset
dataset = pd.read_csv("50_Startups.csv")

# Convert categorical data to numerical
dataset = pd.get_dummies(dataset, drop_first=True)

# Input and output split
independent = dataset[
    [
        'R&D Spend',
        'Administration',
        'Marketing Spend',
        'State_Florida',
        'State_New York'
    ]
]

dependent = dataset[['Profit']]

# Train-test split
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    independent,
    dependent,
    test_size=0.30,
    random_state=0
)

# Model creation
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

# Train model
regressor.fit(X_train, Y_train)

# Coefficients
weight = regressor.coef_
print("Weights:", weight)

# Intercept
bias = regressor.intercept_
print("Bias:", bias)

# Prediction
Y_pred = regressor.predict(X_test)

# Evaluation
from sklearn.metrics import r2_score

r_score = r2_score(Y_test, Y_pred)
print("R2 Score:", r_score)

# Save model
import pickle

filename = "finalized_model_multiple_linear.sav"

pickle.dump(regressor, open(filename, 'wb'))

# Load model
loaded_model = pickle.load(open(filename, 'rb'))

# Test prediction
result = loaded_model.predict([[1234, 345, 346, 1, 0]])

print("Prediction:", result)