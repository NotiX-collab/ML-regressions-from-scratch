import matplotlib.pyplot as plt
import pandas as pd
from model import linear_compute_cost

# Load and clean testing data
df = pd.read_csv("test.csv")
df["x"] = pd.to_numeric(df["x"], errors="coerce", downcast="float")
df["y"] = pd.to_numeric(df["y"], errors="coerce", downcast="float")
df = df.dropna()
X_test= df["x"].values
Y_test = df["y"].values

# Load weights and bias from the file
with open("weights.txt", "r") as file:
    w, b = [float(x) for x in file]

# Calculate the final Mean Squared Error on the test set
# Using the function imported from model.py
MSE_test = linear_compute_cost(X_test, Y_test, w, b)
# Initialize model
Model = w * X_test + b

# Generate testing report visualization
plt.figure(figsize=(10, 5))
plt.scatter(X_test, Y_test, color="blue", alpha=0.4, label="Test Data")
plt.plot(X_test, Model, color="red", linewidth=2, label="Predict")
plt.title(f"Final MSE: {MSE_test:.4f}", fontsize=16)
plt.grid(True, alpha=0.2)
plt.legend()
plt.show()
