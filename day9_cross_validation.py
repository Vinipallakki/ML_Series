# ==================================================
# DAY 9: TRAIN-TEST SPLIT AND CROSS-VALIDATION
# ==================================================

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ==================================================
# 1. CREATE SAMPLE MANUFACTURING DATA
# ==================================================

np.random.seed(42)

n = 1000

df = pd.DataFrame({
    "station": np.random.choice(
        ["Station_A", "Station_B", "Station_C"],
        size=n
    ),
    "temperature": np.random.normal(75, 10, n),
    "standard_cycle_time": np.random.uniform(30, 120, n),
    "queue_length": np.random.randint(0, 30, n)
})

# Simulate overrun probability using available features
logit = (
    -4.0
    + 0.035 * (df["temperature"] - 70)
    + 0.08 * df["queue_length"]
    + 0.025 * (df["standard_cycle_time"] - 60)
    + df["station"].map({
        "Station_A": 0.0,
        "Station_B": 0.4,
        "Station_C": 0.8
    })
)

probability = 1 / (1 + np.exp(-logit))

df["overrun"] = np.random.binomial(1, probability)

print("\nDataset preview:")
print(df.head())

print("\nDataset shape:", df.shape)

print("\nTarget distribution:")
print(df["overrun"].value_counts())


# ==================================================
# 2. SEPARATE FEATURES AND TARGET
# ==================================================

X = df[
    [
        "station",
        "temperature",
        "standard_cycle_time",
        "queue_length"
    ]
]

y = df["overrun"]


# ==================================================
# 3. TRAIN-TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ==================================================
# 4. CREATE PREPROCESSING AND MODEL PIPELINE
# ==================================================

categorical_features = ["station"]

numerical_features = [
    "temperature",
    "standard_cycle_time",
    "queue_length"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)


# ==================================================
# 5. TRAIN THE MODEL
# ==================================================

model.fit(X_train, y_train)

print("\nModel training completed.")


# ==================================================
# 6. EVALUATE ON TRAINING DATA
# ==================================================

train_predictions = model.predict(X_train)

train_accuracy = accuracy_score(
    y_train,
    train_predictions
)

print("\nTraining Accuracy:", round(train_accuracy, 4))


# ==================================================
# 7. EVALUATE ON UNSEEN TEST DATA
# ==================================================

test_predictions = model.predict(X_test)

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

test_precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0
)

test_recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0
)

test_f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0
)

print("\n========== TEST RESULTS ==========")
print("Accuracy:", round(test_accuracy, 4))
print("Precision:", round(test_precision, 4))
print("Recall:", round(test_recall, 4))
print("F1 Score:", round(test_f1, 4))


# ==================================================
# 8. FIVE-FOLD CROSS-VALIDATION
# ==================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring="accuracy"
)

print("\n========== CROSS-VALIDATION ==========")

for fold, score in enumerate(cv_scores, start=1):
    print(f"Fold {fold} accuracy: {score:.4f}")

print("Average CV accuracy:", round(cv_scores.mean(), 4))
print("CV score standard deviation:", round(cv_scores.std(), 4))


# ==================================================
# 9. COMPARE TRAINING AND TEST PERFORMANCE
# ==================================================

print("\n========== PERFORMANCE COMPARISON ==========")

print("Training accuracy:", round(train_accuracy, 4))
print("Test accuracy:", round(test_accuracy, 4))
print("Average CV accuracy:", round(cv_scores.mean(), 4))

if train_accuracy - test_accuracy > 0.10:
    print("Warning: Investigate possible overfitting.")
else:
    print("No large training-test accuracy gap detected.")


# ==================================================
# DAY 9 COMPLETED
# ==================================================

print("\nDay 9 completed successfully!")