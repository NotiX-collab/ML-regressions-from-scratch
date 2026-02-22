import numpy as np
import time
import matplotlib.pyplot as plt


# Implementing linear regressions using loops
class LinearRegressionNaive:
    def __init__(self, alpha=0.001, iterations=1000):
        self.alpha = alpha
        self.iterations = iterations
        self.w = None
        self.b = 0.0
        self.history = []

    # Calculate the Mean Squared Error (MSE) cost function for multiple features
    def compute_cost(self, X, Y):
        m, n = X.shape
        total_cost = 0
        for i in range(m):
            # Inner loop to calculate prediction: f_wb = w1*x1 + w2*x2 + ... + b
            f_wb_i = 0
            for j in range(n):
                f_wb_i += self.w[j] * X[i, j]
            f_wb_i += self.b

            total_cost += (f_wb_i - Y[i]) ** 2
        return total_cost / (2 * m)

    # Compute the gradients for w (multiple) and b
    def compute_gradient(self, X, y):
        m, n = X.shape
        dj_dw = np.zeros(n)
        dj_db = 0.0

        for i in range(m):
            # Again, inner loop for prediction
            f_wb_i = 0
            for j in range(n):
                f_wb_i += self.w[j] * X[i, j]
            f_wb_i += self.b

            err_i = f_wb_i - y[i]

            # Inner loop to update each gradient dj_dw_j
            for j in range(n):
                dj_dw[j] += err_i * X[i, j]
            dj_db += err_i

        return dj_dw / m, dj_db / m

    def fit(self, X, Y):
        m, n = X.shape
        self.w = np.zeros(n) # Initialize weights for each feature
        self.history = []

        for iteration in range(self.iterations):
            dj_dw, dj_db = self.compute_gradient(X, Y)

            # Update all weights
            for j in range(n):
                self.w[j] -= self.alpha * dj_dw[j]
            self.b -= self.alpha * dj_db

            if iteration % 1000 == 0 or iteration == self.iterations - 1:
                curr_cost = self.compute_cost(X, Y)
                self.history.append(self.compute_cost(X, Y))
                print(f"Iteration {iteration}: Cost {curr_cost}")

        return self.w, self.b, self.history


# Implementing linear regressions using NumPy (Vectorization)
class LinearRegressionVectorized:
    def __init__(self, alpha=0.001, iterations=1000, _lambda=0.1):
        self.alpha = alpha
        self.iterations = iterations
        self._lambda = _lambda
        self.w = None
        self.b = 0.0
        self.mu = None
        self.sigma = None
        self.history = []

    # Normalizes the features using Z-score (Mean and Standard Deviation)
    def z_score_normalize(self, X):
        # Calculate mean and sigma only during training (fit)
        if self.mu is None:
            self.mu = np.mean(X, axis=0)
            self.sigma = np.std(X, axis=0)

        X_norm = (X - self.mu) / (self.sigma + 1e-8)
        return X_norm

    # Calculate the Mean Squared Error (MSE) cost function with L2 regularization
    def compute_cost(self, X, y):
        m = X.shape[0]
        # Calculate vectorized prediction error: (Xw + b) - y
        error = np.dot(X, self.w) + self.b - y

        # MSE + regularization penalty for weights
        reg_cost = (self._lambda / (2 * m)) * np.sum(self.w**2)
        cost = np.sum(error**2) / (2 * m) + reg_cost
        return cost

    # Compute vectorized gradients for weight (w) and bias (b)
    def compute_gradient(self, X, y):
        m = X.shape[0]
        # Predicted values using dot product
        f_wb = np.dot(X, self.w) + self.b
        err = f_wb - y

        # Vectorized gradient calculation with L2 regularization
        dj_dw = (1 / m) * np.dot(X.T, err) + (self._lambda / m) * self.w
        dj_db = (1 / m) * np.sum(err)
        return dj_dw, dj_db

    # Perform Gradient Descent to optimize parameters using vectorization
    def fit(self, X, y):
        # Apply normalization to the input features
        X_norm = self.z_score_normalize(X)

        m, n = X_norm.shape
        # Initialize weights as a NumPy array of zeros for n features
        self.w = np.zeros(n)
        self.b = 0.0
        self.history = []

        for iteration in range(self.iterations):
            dj_dw, dj_db = self.compute_gradient(X_norm, y)

            # Update parameters simultaneously
            self.w -= self.alpha * dj_dw
            self.b -= self.alpha * dj_db

            # Print cost status for tracking progress
            if iteration % 1000 == 0 or iteration == self.iterations - 1:
                curr_cost = self.compute_cost(X_norm, y)
                self.history.append(curr_cost)
                print(f"Iteration {iteration}: Cost {curr_cost:.4f}")

        return self.w, self.b, self.history

    # Predict target values for new data
    def predict(self, X):
        # Normalize incoming data using training mu and sigma
        X_norm = self.z_score_normalize(X)
        return np.dot(X_norm, self.w) + self.b


# --- COMPARISON AND TESTING BLOCK ---
if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.rand(2000, 4)
    # Define "true" weights and bias
    true_w, true_b = np.array([1.5, -2, 3, 0.5]), 10
    y = np.dot(X, true_w) + true_b + np.random.randn(2000) * 0.01

    # Naive test
    m1 = LinearRegressionNaive(alpha=0.01, iterations=10000)
    start = time.time()
    m1.fit(X, y)
    t1 = time.time() - start

    # Vectorized test
    m2 = LinearRegressionVectorized(alpha=0.01, iterations=10000, _lambda=1)
    start = time.time()
    m2.fit(X, y)
    t2 = time.time() - start

    print(f"Naive: {t1:.4f}s | Vectorized: {t2:.4f}s")
    print(f"Speedup: {t1 / t2:.1f}x")
    print(f"Pred (Vec): {m2.predict(X[:1])[0]:.2f} | Real: {y[0]:.2f}")
    print(f"Final Weights: {m2.w} | Final Bias: {m2.b:.4f}")

    # Preparing for visualization
    x_naive = [i * 1000 for i in range(len(m1.history))]
    x_vec = [i * 1000 for i in range(len(m2.history))]

    plt.figure(figsize=(10, 12))

    # Graphic 1: Naive (loops)
    plt.subplot(2, 1, 1)
    plt.plot(x_naive, m1.history, color='#e74c3c', linewidth=2, label="Naive MSE")
    plt.title("Naive Implementation (Loops) Learning Curve", fontsize=14)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.legend(shadow=True)
    plt.grid(True, alpha=0.3)

    # Graphic 2: Vectorized (Matrices)
    plt.subplot(2, 1, 2)
    plt.plot(x_vec, m2.history, color='#2ecc71', linewidth=2, label="Vectorized MSE")
    plt.title("Vectorized Implementation Learning Curve", fontsize=14)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.legend(shadow=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()