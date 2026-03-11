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

# Feature and Target
X = df[['AveRooms']]  # Average Rooms
y = housing.target     # Median House Value

# --------------------------
# SPLIT DATA
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------
# TRAIN LINEAR REGRESSION MODEL
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
print(f"Coefficient: {model.coef_[0]:.4f}")

# --------------------------
# VISUALIZATION
# --------------------------
plt.figure(figsize=(8,5))
plt.scatter(X_test, y_test, color="purple", label="Actual Prices")
plt.plot(X_test, y_pred, color="green", linewidth=2, label="Regression Line")
plt.xlabel("Average Rooms")
plt.ylabel("House Price (in $100,000s)")
plt.title("Simple Linear Regression: House Price vs Average Rooms")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()
