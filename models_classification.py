import numpy as np
import time
import matplotlib.pyplot as plt


# Implementing logistic regression using loops (naive approach)
class LogisticRegressionNaive:
    def __init__(self, alpha=0.01, iterations=1000):
        self.alpha = alpha
        self.iterations = iterations
        self.w = None
        self.b = 0.0
        self.history = []

    # Sigmoid activation function to map values to [0, 1]
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # Calculate Binary Cross-Entropy loss using nested loops
    def compute_cost(self, X, y):
        m, n = X.shape
        total_cost = 0

        for i in range(m):
            # Manual dot product for the prediction
            z_i = 0
            for j in range(n):
                z_i += self.w[j] * X[i, j]
            z_i += self.b
            f_wb_i = self._sigmoid(z_i)

            # Clip values to avoid log(0)
            f_wb_i = max(min(f_wb_i, 1 - 1e-15), 1e-15)
            total_cost += -(y[i] * np.log(f_wb_i) + (1 - y[i]) * np.log(1 - f_wb_i))

        return total_cost / m

    # Compute the gradients for w (multiple) and b
    def compute_gradient(self, X, y):
        m, n = X.shape
        dj_dw = np.zeros(n)
        dj_db = 0.0

        for i in range(m):
            # Prediction for the i-th sample
            z_i = 0
            for j in range(n):
                z_i += self.w[j] * X[i, j]
            z_i += self.b
            f_wb_i = self._sigmoid(z_i)

            err_i = f_wb_i - y[i]

            # Update gradient for each feature
            for j in range(n):
                dj_dw[j] += err_i * X[i, j]
            dj_db += err_i

        return dj_dw / m, dj_db / m

    def fit(self, X, y):
        m, n = X.shape
        self.w = np.zeros(n)  # Initialize weights for each feature

        for iteration in range(self.iterations):
            dj_dw, dj_db = self.compute_gradient(X, y)

            # Update parameters
            for j in range(n):
                self.w[j] -= self.alpha * dj_dw[j]
            self.b -= self.alpha * dj_db

            if iteration % 1000 == 0:
                cost = self.compute_cost(X, y)
                self.history.append(cost)
                print(f"Iteration {iteration}: Cost {cost:.4f}")


# Optimized Logistic Regression using NumPy vectorization and L2 regularization
class LogisticRegressionVectorized:
    def __init__(self, alpha=0.01, iterations=1000, _lambda=0.1):
        self.alpha = alpha
        self.iterations = iterations
        self._lambda = _lambda  # Regularization parameter
        self.w = None
        self.b = 0.0
        self.mu = None
        self.sigma = None
        self.history = []

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def z_score_normalize(self, X):
        # Calculate mean and sigma only during training (fit)
        if self.mu is None:
            self.mu = np.mean(X, axis=0)
            self.sigma = np.std(X, axis=0)

        X_norm = (X - self.mu) / (self.sigma + 1e-8)
        return X_norm

    # Vectorized Log Loss with L2 regularization
    def compute_cost(self, X, y):
        m = X.shape[0]
        z = np.dot(X, self.w) + self.b
        f_wb = self._sigmoid(z)

        # Clip to prevent numerical instability
        f_wb = np.clip(f_wb, 1e-15, 1 - 1e-15)

        log_loss = -(y * np.log(f_wb) + (1 - y) * np.log(1 - f_wb))
        cost = np.sum(log_loss) / m

        # Add L2 Regularization
        reg_cost = (self._lambda / (2 * m)) * np.sum(self.w**2)
        return cost + reg_cost

    # Compute gradients using matrix operations
    def compute_gradient(self, X, y):
        m = X.shape[0]
        f_wb = self._sigmoid(np.dot(X, self.w) + self.b)
        err = f_wb - y

        # Vectorized gradient calculation
        dj_dw = (1 / m) * np.dot(X.T, err) + (self._lambda / m) * self.w
        dj_db = (1 / m) * np.sum(err)

        return dj_dw, dj_db

    # Train the model efficiently using NumPy
    def fit(self, X, y):
        m, n = X.shape
        self.w = np.zeros(n)

        for i in range(self.iterations):
            dj_dw, dj_db = self.compute_gradient(X, y)

            self.w -= self.alpha * dj_dw
            self.b -= self.alpha * dj_db

            if i % 1000 == 0:
                cost = self.compute_cost(X, y)
                self.history.append(cost)
                print(f"Vectorized Iteration {i}: Cost {cost:.4f}")

    # Predict target values for new data
    def predict(self, X):
        prob = self._sigmoid(np.dot(X, self.w) + self.b)
        return (prob >= 0.5).astype(int)


# --- COMPARISON AND TESTING BLOCK ---
if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.randn(1000, 4)
    true_w = np.array([3.5, -2.5, 1.0, -0.5])
    true_b = 0.5

    # Generate binary labels (0 or 1)
    z = np.dot(X, true_w) + true_b
    probabilities = 1 / (1 + np.exp(-z))
    y = (probabilities > 0.5).astype(int)

    # Naive test
    m1 = LogisticRegressionNaive(alpha=0.01, iterations=10000)
    start = time.time()
    m1.fit(X, y)
    t1 = time.time() - start

    # Vectorized test
    m2 = LogisticRegressionVectorized(alpha=0.01, iterations=10000, _lambda=1)
    start = time.time()
    m2.fit(X, y)
    t2 = time.time() - start

    print(f"Naive: {t1:.4f}s | Vectorized: {t2:.4f}s")
    print(f"Speedup: {t1 / t2:.1f}x")
    print(f"Pred (Vec): {m2.predict(X[:1])[0]:.2f} | Real: {y[0]:.2f}")
    print(f"Final Weights: {m2.w} | Final Bias: {m2.b:.4f}")

    # Visualization
    x_axis = [i * 1000 for i in range(len(m1.history))]

    plt.figure(figsize=(10, 12))

    # Plot 1: Naive Implementation
    plt.subplot(2, 1, 1)
    plt.plot(x_axis, m1.history, color="#e74c3c", linewidth=2, label="Naive LogLoss")
    plt.title("Naive Implementation (Loops) Learning Curve", fontsize=14)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.legend(shadow=True)
    plt.grid(True, alpha=0.3)

    # Plot 2: Vectorized Implementation
    plt.subplot(2, 1, 2)
    plt.plot(
        x_axis, m2.history, color="#2ecc71", linewidth=2, label="Vectorized LogLoss"
    )
    plt.title("Vectorized Implementation Learning Curve", fontsize=14)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.legend(shadow=True)
    plt.grid(True, alpha=0.3)

    plt.show()

