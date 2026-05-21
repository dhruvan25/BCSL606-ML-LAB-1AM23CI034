# Adaptive Boost Regression - Insurance Charges Prediction

# Import libraries
import pandas as pd

# Load dataset
dataset = pd.read_csv("insurance_pre.csv")

# Convert categorical columns into numerical columns
dataset = pd.get_dummies(dataset, drop_first=True)

# Print columns
print("Dataset Columns:")
print(dataset.columns)

# Split input and output
X = dataset.drop("charges", axis=1)
y = dataset["charges"]

# Train-test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=0
)

# Import AdaBoost
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

# Create base estimator
base_model = DecisionTreeRegressor(max_depth=4)

# Create AdaBoost model
model = AdaBoostRegressor(
    estimator=base_model,
    n_estimators=100,
    random_state=0
)

# Train model
model.fit(X_train, y_train)

# Prediction on test data
y_pred = model.predict(X_test)

# Evaluation
from sklearn.metrics import r2_score

score = r2_score(y_test, y_pred)

print("\nR2 Score:", score)

# Print feature columns used for training
print("\nFeatures used for training:")
print(X.columns)

# Example sample prediction
# Change values according to printed columns order

sample = [[
    19,     # age
    27.9,   # bmi
    0,      # children
    1,      # sex_male
    0       # smoker_yes
]]

# Predict insurance charge
prediction = model.predict(sample)

print("\nPredicted Insurance Charge:")
print(prediction)