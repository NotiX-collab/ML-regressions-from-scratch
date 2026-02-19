import pandas as pd
import numpy as np
import pickle


test_data = pd.read_csv("test.csv")
test_ids = test_data["Id"]

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

test_data["Fence"] = test_data["Fence"].fillna("None")
test_data["GarageType"] = test_data["GarageType"].fillna("None")
test_data["Functional"] = test_data["Functional"].fillna("Typ")
test_data["GarageFinish"] = test_data["GarageFinish"].fillna("None")
test_data["Electrical"] = test_data["Electrical"].fillna(
    test_data["Electrical"].mode()[0]
)

for col in qual_cols:
    if col in test_data.columns:
        test_data[col] = test_data[col].map(mapping).fillna(0)

cat_cols = [
    "Neighborhood",
    "MSZoning",
    "Foundation",
    "Fence",
    "GarageType",
    "GarageFinish",
]

test_data = pd.get_dummies(test_data, columns=cat_cols)
test_data = test_data.select_dtypes(exclude=["object"])

with open('house_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

X_test_raw = test_data.reindex(columns=model_data['features'], fill_value=0)
X_test_raw = X_test_raw.fillna(0)
X_test_norm = (X_test_raw.values - model_data['mu']) / model_data['sigma']
y_final_pred = np.dot(X_test_norm, model_data['weights']) + model_data['bias']

submission = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": y_final_pred
})

submission.to_csv("submission.csv", index=False)