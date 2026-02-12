import matplotlib.pyplot as plt
import pandas as pd
from classification_model import classification_compute_cost, sigmoid

# Load and prepare the TEST dataset (Unseen data)
df = pd.read_csv("student_data_test.csv")

# Clean data
df["x"] = pd.to_numeric(df["hours"], errors="coerce", downcast="float")
df["y"] = pd.to_numeric(df["pass"], errors="coerce", downcast="float")
df = df.dropna()
X_test = df["x"].values
Y_test = df["y"].values

# Load the trained model parameters (Weights & Bias)
with open("weights_1.txt", "r") as file:
    w, b = [float(x) for x in file]

# Calculate the Loss (Cost) on test data
log_loss = classification_compute_cost(X_test, Y_test, w, b)

# Visualization
plt.figure(figsize=(12, 6))

# Scatter plot of actual test data
plt.scatter(X_test, Y_test, color="blue", alpha=0.3, label="Test Data (Actual)")

# Decision Boundary (Threshold 0.5)
plt.axhline(y=0.5, color='gray', linestyle='--', label='Decision Threshold (0.5)')

# Generate a smooth sigmoid curve using 100 points based on our learned weights
x_line = [i * 0.1 for i in range(101)]  # 0.0 to 10.0
y_line = [sigmoid(w * x + b) for x in x_line]

# Plot the predicted probability curve
plt.plot(x_line, y_line, color="red", linewidth=3, label="Logistic Model (Predicted)")
plt.title(f"Model Performance on Test Data (Loss: {log_loss:.1f}%)")
plt.xlabel("Hours")
plt.ylabel("Pass")
plt.legend()
plt.grid(True, alpha=0.2)
plt.show()