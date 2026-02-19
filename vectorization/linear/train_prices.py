import pandas as pd
import numpy as np
import pickle
from vectorization_linear import (
    gradient_descent,
    compute_gradient,
    compute_cost,
    z_score_normalize,
)


train_data = pd.read_csv("train.csv")

mapping = {
    "Ex": 5,
    "Gd": 4,
    "TA": 3,
    "Fa": 2,
    "Po": 1,
    "None": 0,
    "Fin": 3,
    "RFN": 2,
    "Unf": 1,
}
qual_cols = [
    "ExterQual",
    "KitchenQual",
    "HeatingQC",
    "BsmtQual",
    "FireplaceQu",
    "GarageQual",
    "GarageCond",
]

train_data["Fence"] = train_data["Fence"].fillna("None")
train_data["GarageType"] = train_data["GarageType"].fillna("None")
train_data["Functional"] = train_data["Functional"].fillna("Typ")
train_data["GarageFinish"] = train_data["GarageFinish"].fillna("None")
train_data["Electrical"] = train_data["Electrical"].fillna(
    train_data["Electrical"].mode()[0]
)

for col in qual_cols:
    if col in train_data.columns:
        train_data[col] = train_data[col].map(mapping).fillna(0)

cat_cols = [
    "Neighborhood",
    "MSZoning",
    "Foundation",
    "Fence",
    "GarageType",
    "GarageFinish",
]
train_data = pd.get_dummies(train_data, columns=cat_cols)

train_data = train_data.select_dtypes(exclude=["object"])
numeric_data = train_data.select_dtypes(include=[np.number])
feature_names = numeric_data.drop(
    ["Id", "SalePrice"], axis=1, errors="ignore"
).columns.tolist()
X_raw = numeric_data.drop(["Id", "SalePrice"], axis=1, errors="ignore")
y_train = numeric_data["SalePrice"].values
X_raw = X_raw.fillna(X_raw.mean()).values
X_train, mu_train, sigma_train = z_score_normalize(X_raw)

# Parameters
m, n = X_train.shape
initial_w = np.zeros(n)
initial_b = 0.0
alpha = 0.001
_lambda = 1

iterations = 50000

w_final, b_final, J_history = gradient_descent(
    X_train,
    y_train,
    initial_w,
    initial_b,
    compute_cost,
    compute_gradient,
    _lambda,
    alpha,
    iterations,
)


print(f"После обучения: {w_final}, {b_final}, {J_history[-1]:.2f}")

model_data = {
    "weights": w_final,
    "bias": b_final,
    "mu": mu_train,
    "sigma": sigma_train,
    "features": feature_names,
}

with open("house_model.pkl", "wb") as weights:
    pickle.dump(model_data, weights)

