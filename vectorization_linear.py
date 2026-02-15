import numpy as np
import pandas as pd


def compute_cost(X, y, w, b):
    m = X.shape[0]
    error = np.dot(X, w) + b - y
    cost = np.sum(error**2) / (2 * m)
    return cost


def compute_gradient(X, y, w, b):
    m = X.shape[0]
    f_wb = np.dot(X, w) + b
    err = f_wb - y
    dj_dw = (1 / m) * np.dot(X.T, err)
    dj_db = (1 / m) * np.sum(err)
    return dj_dw, dj_db


def gradient_descent(
    X, y, w, b, compute_cost, compute_gradient, alpha=0.0001, iterations=10000
):
    history = []
    for i in range(iterations):
        dj_dw, dj_db = compute_gradient(X, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db

        if i % 100 == 0 or i == iterations - 1:
            curr_cost = compute_cost(X, y, w, b)
            history.append(curr_cost)

            if len(history) > 1 and abs(history[-2] - history[-1]) < 1e-6:
                break

    return w, b, history
