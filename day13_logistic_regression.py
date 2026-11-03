
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ==================================================
# 1. CREATE MANUFACTURING DATASET
# ==================================================

np.random.seed(42)

n = 1000

df = pd.DataFrame({
    "temperature": np.random.normal(75, 10, n),
    "queue_length": np.random.randint(0, 30, n),
    "standard_cycle_time": np.random.uniform(30, 120, n),
    "station": np.random.choice(
        ["Station_A", "Station_B", "Station_C"], n
    ),
    "actual_cycle_time": np.random.uniform(30, 150, n)
})

# Create target: 1 = Overrun, 0 = No Overrun
df["overrun"] = (
    df["actual_cycle_time"] > df["standard_cycle_time"]
).astype(int)

print("\n========== DATASET ==========")
print(df.head())

print("\nTarget distribution:")
print(df["overrun"].value_counts())

# ==================================================
# 2. SEPARATE INPUT FEATURES AND TARGET
# ==================================================

X = df.drop(
    ["overrun", "actual_cycle_time"],
    axis=1
)

y = df["overrun"]

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())

# ==================================================
# 3. TRAIN-TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# ==================================================
# 4. PREPROCESSING
# ==================================================

numerical_features = [
    "temperature",
    "queue_length",
    "standard_cycle_time"
]

categorical_features = ["station"]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            StandardScaler(),
            numerical_features
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)

# ==================================================
# 5. CREATE LOGISTIC REGRESSION MODEL
# ==================================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(max_iter=1000)
        )
    ]
)

# ==================================================
# 6. TRAIN MODEL
# ==================================================

model.fit(X_train, y_train)

print("\nModel training completed!")

# ==================================================
# 7. PREDICTION
# ==================================================

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)

print("\n========== FIRST 10 PREDICTIONS ==========")

for i in range(10):
    print(
        f"Actual: {y_test.iloc[i]} | "
        f"Predicted: {y_pred[i]} | "
        f"Probability: {y_probability[i]}"
    )

# ==================================================
# 8. MODEL EVALUATION
# ==================================================

print("\n========== MODEL EVALUATION ==========")

print("Accuracy:", accuracy_score(y_test, y_pred))

print("Precision:", precision_score(y_test, y_pred))

print("Recall:", recall_score(y_test, y_pred))

print("F1 Score:", f1_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==================================================
# 9. UNDERSTAND LEARNED COEFFICIENTS
# ==================================================

lr_model = model.named_steps["classifier"]

feature_names = (
    model.named_steps["preprocessor"]
    .get_feature_names_out()
)

print("\n========== LEARNED COEFFICIENTS ==========")

for feature, coefficient in zip(
    feature_names,
    lr_model.coef_[0]
):
    print(f"{feature}: {coefficient:.4f}")

print("Intercept:", lr_model.intercept_[0])

# ==================================================
# 10. PREDICT A NEW MACHINE OPERATION
# ==================================================

new_operation = pd.DataFrame({
    "temperature": [85],
    "queue_length": [20],
    "standard_cycle_time": [60],
    "station": ["Station_B"]
})

new_probability = model.predict_proba(
    new_operation
)[0]

new_prediction = model.predict(new_operation)[0]

print("\n========== NEW MACHINE PREDICTION ==========")

print("No Overrun Probability:", round(new_probability[0], 4))
print("Overrun Probability:", round(new_probability[1], 4))

print("Predicted Class:", new_prediction)

if new_prediction == 1:
    print("Result: Machine operation may exceed standard time.")
else:
    print("Result: Machine operation is expected within standard time.")

# ==================================================
# 11. CUSTOM THRESHOLD EXPERIMENT
# ==================================================

print("\n========== THRESHOLD EXPERIMENT ==========")

overrun_probabilities = model.predict_proba(X_test)[:, 1]

for threshold in [0.3, 0.5, 0.7]:

    custom_predictions = (
        overrun_probabilities >= threshold
    ).astype(int)

    print(f"\nThreshold: {threshold}")
    print(
        "Precision:",
        round(precision_score(y_test, custom_predictions), 4)
    )
    print(
        "Recall:",
        round(recall_score(y_test, custom_predictions), 4)
    )
    print(
        "F1 Score:",
        round(f1_score(y_test, custom_predictions), 4)
    )
