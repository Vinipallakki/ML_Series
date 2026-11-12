import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# ------------------------------------------------
# 1. Generate manufacturing data
# ------------------------------------------------

np.random.seed(42)
n = 2000

df = pd.DataFrame({
    "temperature": np.random.normal(75, 10, n),
    "vibration": np.random.uniform(1, 10, n),
    "pressure": np.random.uniform(100, 200, n),
    "machine_age": np.random.uniform(1, 15, n),
    "station": np.random.choice(
        ["Station_A", "Station_B", "Station_C"], n
    )
})

# Station-specific effect
station_effect = df["station"].map({
    "Station_A": 0.0,
    "Station_B": 0.4,
    "Station_C": 0.8
})

# ------------------------------------------------
# 2. Generate defect target
# ------------------------------------------------

score = (
    -4
    + 0.05 * (df["temperature"] - 70)
    + 0.40 * df["vibration"]
    + 0.02 * (df["pressure"] - 150)
    + 0.08 * df["machine_age"]
    + station_effect
)

probability = 1 / (1 + np.exp(-score))

df["defect"] = np.random.binomial(1, probability)

print("Dataset:")
print(df.head())

print("\nTarget distribution:")
print(df["defect"].value_counts())

# ------------------------------------------------
# 3. Separate features and target
# ------------------------------------------------

X = df.drop("defect", axis=1)
y = df["defect"]

# ------------------------------------------------
# 4. Train-test split
# ------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ------------------------------------------------
# 5. LightGBM preparation
# ------------------------------------------------

# Tell LightGBM that station is categorical
X_train_lgb = X_train.copy()
X_test_lgb = X_test.copy()

categories = ["Station_A", "Station_B", "Station_C"]

X_train_lgb["station"] = pd.Categorical(
    X_train_lgb["station"], categories=categories
)

X_test_lgb["station"] = pd.Categorical(
    X_test_lgb["station"], categories=categories
)

# ------------------------------------------------
# 6. Create models
# ------------------------------------------------

lgbm_model = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    random_state=42,
    verbosity=-1
)

catboost_model = CatBoostClassifier(
    iterations=100,
    learning_rate=0.05,
    depth=6,
    loss_function="Logloss",
    verbose=0,
    random_seed=42
)

# ------------------------------------------------
# 7. Train models
# ------------------------------------------------

print("\nTraining LightGBM...")
lgbm_model.fit(X_train_lgb, y_train)

print("Training CatBoost...")
catboost_model.fit(
    X_train,
    y_train,
    cat_features=["station"]
)

# ------------------------------------------------
# 8. Evaluate both models
# ------------------------------------------------

models = {
    "LightGBM": (lgbm_model, X_test_lgb),
    "CatBoost": (catboost_model, X_test)
}

for name, (model, test_data) in models.items():

    print(f"\n========== {name} ==========")

    y_pred = model.predict(test_data)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# ------------------------------------------------
# 9. Predict a new machine operation
# ------------------------------------------------

new_operation = pd.DataFrame({
    "temperature": [90],
    "vibration": [8.5],
    "pressure": [180],
    "machine_age": [10],
    "station": ["Station_B"]
})

# LightGBM prediction
new_operation_lgb = new_operation.copy()

new_operation_lgb["station"] = pd.Categorical(
    new_operation_lgb["station"],
    categories=categories
)

lgb_prediction = lgbm_model.predict(new_operation_lgb)
lgb_probability = lgbm_model.predict_proba(new_operation_lgb)

# CatBoost prediction
cat_prediction = catboost_model.predict(new_operation)
cat_probability = catboost_model.predict_proba(new_operation)

print("\nNew Operation Predictions:")

print("\nLightGBM:")
print("Predicted Defect:", lgb_prediction[0])
print("Defect Probability:", lgb_probability[0][1])

print("\nCatBoost:")
print("Predicted Defect:", cat_prediction[0])
print("Defect Probability:", cat_probability[0][1])