import numpy as np


def z_score_normalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / (sigma + 1e-8)
    return X_norm, mu, sigma


def compute_cost(X, y, w, b, _lambda):
    m = X.shape[0]
    error = np.dot(X, w) + b - y
    cost = np.sum(error**2) / (2 * m) + (_lambda / (2 * m)) * np.sum(w ** 2)
    return cost


def compute_gradient(X, y, w, b, _lambda):
    m = X.shape[0]
    f_wb = np.dot(X, w) + b
    err = f_wb - y
    dj_dw = (1 / m) * np.dot(X.T, err) + (_lambda / m) * w
    dj_db = (1 / m) * np.sum(err)
    return dj_dw, dj_db


def gradient_descent(
    X, y, w, b, compute_cost, compute_gradient,  _lambda, alpha=0.0001, iterations=10000
):
    history = []
    for i in range(iterations):
        dj_dw, dj_db = compute_gradient(X, y, w, b, _lambda)
        w -= alpha * dj_dw
        b -= alpha * dj_db

        if i % 1000 == 0 or i == iterations - 1:
            curr_cost = compute_cost(X, y, w, b, _lambda)
            history.append(curr_cost)
            print(f"Итерация {i}: Cost {curr_cost:.2f}")

    return w, b, history
