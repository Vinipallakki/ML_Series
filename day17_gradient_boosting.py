import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# 1. Generate manufacturing data
np.random.seed(42)
n = 2000

df = pd.DataFrame({
    "temperature": np.random.normal(75, 10, n),
    "vibration": np.random.uniform(1, 10, n),
    "pressure": np.random.uniform(100, 200, n),
    "machine_age": np.random.uniform(1, 15, n)
})

# 2. Generate defect target
score = (
    -4
    + 0.05 * (df["temperature"] - 70)
    + 0.40 * df["vibration"]
    + 0.02 * (df["pressure"] - 150)
    + 0.08 * df["machine_age"]
)

probability = 1 / (1 + np.exp(-score))

df["defect"] = np.random.binomial(1, probability)

# 3. Separate features and target
X = df.drop("defect", axis=1)
y = df["defect"]

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. Create models
models = {
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
}

# 6. Train and evaluate
for name, model in models.items():

    print(f"\n========== {name} ==========")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# 7. Predict a new machine operation
new_operation = pd.DataFrame({
    "temperature": [90],
    "vibration": [8.5],
    "pressure": [180],
    "machine_age": [10]
})

xgb_model = models["XGBoost"]

prediction = xgb_model.predict(new_operation)
probability = xgb_model.predict_proba(new_operation)

print("\nNew Operation Prediction:")
print("Predicted Defect:", prediction[0])
print("No Defect Probability:", probability[0][0])
print("Defect Probability:", probability[0][1])

# 8. Feature importance
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": xgb_model.feature_importances_
}).sort_values("Importance", ascending=False)

print("\nFeature Importance:")
print(importance)