import numpy as np
import pandas as pd
import os

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


base_dir = os.path.dirname(__file__)
train_path = os.path.join(base_dir, "dataset", "train_dataset.csv")
test_path = os.path.join(base_dir, "..", "tests", "test_dataset.csv")  # 🔁 fixed path


train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)


X_train = train_df["feature_0"].values.reshape(-1, 1)
y_train = train_df["target"].values.reshape(-1, 1)
X_test = test_df["feature_0"].values.reshape(-1, 1)
y_test = test_df["target"].values.reshape(-1, 1)


w = 0.0
b = 0.0
alpha = 0.01
epochs = 1000
m = len(X_train)


for epoch in range(epochs):
    y_pred = w * X_train + b
    error = y_pred - y_train
    dw = (1 / m) * np.sum(error * X_train)
    db = (1 / m) * np.sum(error)
    w -= alpha * dw
    b -= alpha * db


y_test_pred = w * X_test + b
mse = mean_squared_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

print("\n✅ Linear Regression (NumPy)")
print(f"Weight (w): {w:.4f}")
print(f"Bias   (b): {b:.4f}")
print(f"Test MSE:   {mse:.4f}")
print(f"Test R²:    {r2:.4f}")
