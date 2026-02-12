import matplotlib.pyplot as plt
import pandas as pd


# Calculate the Mean Squared Error (MSE) cost function
def linear_compute_cost(X, Y, w, b):
    total_cost = 0
    m = X.shape[0]

    for i in range(m):
        total_cost += ((w * X[i] + b) - Y[i]) ** 2

    return total_cost / (2 * m)


# Compute the gradients for w and b
def linear_gradient_cost(X, Y, w, b):
    dj_dw, dj_db = 0, 0
    m = X.shape[0]
    for i in range(m):
        dj_dw += ((w * X[i] + b) - Y[i]) * X[i]
        dj_db += (w * X[i] + b) - Y[i]

    return dj_dw / m, dj_db / m


# Perform Gradient Descent to optimize parameters w and b
def gradient_descent(
    X, Y, w, b, gradient_cost, compute_cost, alpha=0.00001, iterations=10000
):
    prev_cost = 0
    history = []

    for iteration in range(iterations):
        dj_dw, dj_db = gradient_cost(X, Y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db
        cost = compute_cost(X, Y, w, b)
        history.append(cost)

        if iteration % 100 == 0:
            print(f"Iteration: {iteration}, cost: {cost}")

        # Check for convergence (if cost change is negligible)
        if abs(prev_cost - cost) < 1e-6:
            print(f"Iteration: {iteration}, cost: {cost}")
            break

        prev_cost = cost

    return w, b, history


if __name__ == "__main__":
    df = pd.read_csv("train.csv")
    # Load and clean training data
    df["x"] = pd.to_numeric(df["x"], errors="coerce", downcast="float")
    df["y"] = pd.to_numeric(df["y"], errors="coerce", downcast="float")
    df = df.dropna()
    X = df["x"].values
    Y = df["y"].values

    # Initialize model parameters
    w = 0
    b = 0

    # Start training process
    w_train, b_train, history = gradient_descent(
        X, Y, w, b, linear_gradient_cost, linear_compute_cost
    )

    # Save trained weights and bias to a file
    with open("weights.txt", "w") as f:
        f.write(f"{w_train}\n{b_train}")

    # Generate training report visualization
    plt.figure(figsize=(12, 10))

    # Plot 1: Cost Function History
    plt.subplot(2, 1, 1)
    plt.plot(history, linewidth=2, label="Cost function")
    plt.legend(frameon=True, shadow=True, fontsize='medium', facecolor='white')
    plt.xlabel("Iteration")
    plt.ylabel("MSE")
    plt.grid(True)

    # Plot 2: Model Fil on Training Data
    plt.subplot(2, 1, 2)
    plt.scatter(X, Y, color="blue", alpha=0.4, label="Train Data")
    Model = w_train * X + b_train
    plt.plot(X, Model, color="red", linewidth=2, label="Model")
    plt.legend(frameon=True, shadow=True, fontsize='medium', facecolor='white')
    plt.savefig("Cost_function.png", dpi=100)
