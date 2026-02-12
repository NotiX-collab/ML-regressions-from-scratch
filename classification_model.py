import matplotlib.pyplot as plt
import pandas as pd
import numpy as np # Only for np.exp() for accuracy


# Sigmoid function: maps any input into a value between 0 and 1
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Calculate the Binary Cross-Entropy (Log loss) cost function
def classification_compute_cost(X, Y, w, b):
    total_cost = 0
    m = X.shape[0]

    for i in range(m):
        f_wb = sigmoid(w * X[i] + b)
        total_cost += -(Y[i] * np.log(f_wb) + (1 - Y[i]) * np.log(1 - f_wb))

    return total_cost / m


# Compute the gradients for w and b
def classification_gradient_cost(X, Y, w, b):
    dj_dw, dj_db = 0, 0
    m = X.shape[0]
    for i in range(m):
        f_wb = sigmoid(w * X[i] + b)
        dj_dw += (f_wb - Y[i]) * X[i]
        dj_db += f_wb - Y[i]

    return dj_dw / m, dj_db / m


# Perform Gradient Descent to optimize parameters w and b
def gradient_descent(
    X, Y, w, b, gradient_cost, compute_cost, alpha=0.0001, iterations=10000
):
    prev_cost = 0
    history = []

    for iteration in range(iterations):
        dj_dw, dj_db = gradient_cost(X, Y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db
        cost = compute_cost(X, Y, w, b)
        history.append(cost)

        if iteration % 1000 == 0:
            print(f"Iteration: {iteration}, cost: {cost}")

        if abs(prev_cost - cost) < 1e-7:
            print(f"Iteration: {iteration}, cost: {cost}")
            break

        prev_cost = cost

    return w, b, history


if __name__ == "__main__":
    df = pd.read_csv("student_data.csv")
    # Load and clean training data
    df["x"] = pd.to_numeric(df["hours"], errors="coerce", downcast="float")
    df["y"] = pd.to_numeric(df["pass"], errors="coerce", downcast="float")
    df = df.dropna()
    X = df["x"].values
    Y = df["y"].values

    # Initialize model parameters
    w = 0
    b = 0

    # Start training process
    w_train, b_train, history = gradient_descent(
        X, Y, w, b, classification_gradient_cost, classification_compute_cost, alpha=0.1
    )

    # Save trained weights and bias to a file
    with open("weights_1.txt", "w") as f:
        f.write(f"{w_train}\n{b_train}")

    # Visualization
    plt.figure(figsize=(12, 6))

    # Plot 1: Actual Data
    plt.scatter(X, Y, color="blue", alpha=0.3, label="Actual Data (0 or 1)")
    plt.axhline(y=0.5, color='gray', linestyle='--', label='Threshold 0.5') # Threshold

    # Generate smooth sigmoid curve using 100 points
    x_line = [i * 0.1 for i in range(101)]
    y_line = [sigmoid(w_train * x + b_train) for x in x_line]

    # Plot 2: The sigmoid function
    plt.plot(x_line, y_line, color="red", linewidth=3, label="Logistic Regression (Probability)")
    plt.xlabel("Hours")
    plt.ylabel("Pass")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.show()