import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing

# --------------------------
# LOAD DATA
# --------------------------
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)

# Select features and target
X = df[['AveRooms', 'HouseAge', 'Population']]
y = housing.target

# --------------------------
# SPLIT DATA
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------
# TRAIN MULTIPLE LINEAR REGRESSION MODEL
# --------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# --------------------------
# PREDICTION
# --------------------------
y_pred = model.predict(X_test)

# --------------------------
# MODEL EVALUATION
# --------------------------
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}")
print(f"R-Squared Score: {r2_score(y_test, y_pred):.4f}")
print(f"Intercept: {model.intercept_:.4f}")
print(f"Coefficients: {model.coef_}")

# --------------------------
# VISUALIZATION: FEATURE IMPORTANCE
# --------------------------
plt.figure(figsize=(9,5))
plt.bar(X.columns, model.coef_, color='skyblue', edgecolor='black')
plt.xlabel("Features")
plt.ylabel("Coefficient Value")
plt.title("Feature Importance in Multiple Regression")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()
