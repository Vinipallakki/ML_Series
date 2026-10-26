# ==================================================
# DAY 7: FEATURE SCALING AND NORMALIZATION
# ==================================================

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ==================================================
# 1. CREATE SAMPLE DATASET
# ==================================================

data = {
    "temperature": [60, 65, 70, 75, 80, 85, 90, 95, 100, 105],
    "production_volume": [
        1000, 1200, 1500, 1800, 2000,
        2200, 2500, 2800, 3000, 3500
    ],
    "cycle_time": [30, 32, 35, 38, 40, 45, 48, 52, 55, 60],
    "overrun": [0, 0, 0, 0, 0, 1, 0, 1, 1, 1]
}

df = pd.DataFrame(data)

print("\nOriginal Dataset:")
print(df)


# ==================================================
# 2. SEPARATE FEATURES AND TARGET
# ==================================================

X = df[
    ["temperature", "production_volume", "cycle_time"]
]

y = df["overrun"]

print("\nFeatures (X):")
print(X)

print("\nTarget (y):")
print(y)


# ==================================================
# 3. SPLIT TRAINING AND TEST DATA
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))


# ==================================================
# 4. STANDARD SCALER
# ==================================================

standard_scaler = StandardScaler()

# Learn mean and standard deviation from training data
X_train_standard = standard_scaler.fit_transform(X_train)

# Apply the same transformation to test data
X_test_standard = standard_scaler.transform(X_test)

print("\nStandardScaler - Training Data:")
print(
    pd.DataFrame(
        X_train_standard,
        columns=X.columns
    ).round(2)
)


# ==================================================
# 5. MINMAX SCALER
# ==================================================

minmax_scaler = MinMaxScaler()

X_train_minmax = minmax_scaler.fit_transform(X_train)
X_test_minmax = minmax_scaler.transform(X_test)

print("\nMinMaxScaler - Training Data:")
print(
    pd.DataFrame(
        X_train_minmax,
        columns=X.columns
    ).round(2)
)


# ==================================================
# 6. ROBUST SCALER
# ==================================================

robust_scaler = RobustScaler()

X_train_robust = robust_scaler.fit_transform(X_train)
X_test_robust = robust_scaler.transform(X_test)

print("\nRobustScaler - Training Data:")
print(
    pd.DataFrame(
        X_train_robust,
        columns=X.columns
    ).round(2)
)


# ==================================================
# 7. TRAIN MODEL WITHOUT SCALING
# ==================================================

model_original = LogisticRegression(max_iter=1000)

model_original.fit(X_train, y_train)

pred_original = model_original.predict(X_test)

accuracy_original = accuracy_score(
    y_test,
    pred_original
)

print("\nAccuracy Without Scaling:", accuracy_original)


# ==================================================
# 8. TRAIN MODEL WITH STANDARD SCALING
# ==================================================

model_standard = LogisticRegression(max_iter=1000)

model_standard.fit(X_train_standard, y_train)

pred_standard = model_standard.predict(X_test_standard)

accuracy_standard = accuracy_score(
    y_test,
    pred_standard
)

print("Accuracy With StandardScaler:", accuracy_standard)


# ==================================================
# 9. TRAIN MODEL WITH MINMAX SCALING
# ==================================================

model_minmax = LogisticRegression(max_iter=1000)

model_minmax.fit(X_train_minmax, y_train)

pred_minmax = model_minmax.predict(X_test_minmax)

accuracy_minmax = accuracy_score(
    y_test,
    pred_minmax
)

print("Accuracy With MinMaxScaler:", accuracy_minmax)


# ==================================================
# 10. TRAIN MODEL WITH ROBUST SCALING
# ==================================================

model_robust = LogisticRegression(max_iter=1000)

model_robust.fit(X_train_robust, y_train)

pred_robust = model_robust.predict(X_test_robust)

accuracy_robust = accuracy_score(
    y_test,
    pred_robust
)

print("Accuracy With RobustScaler:", accuracy_robust)


# ==================================================
# 11. FINAL COMPARISON
# ==================================================

results = pd.DataFrame({
    "Method": [
        "Without Scaling",
        "StandardScaler",
        "MinMaxScaler",
        "RobustScaler"
    ],
    "Accuracy": [
        accuracy_original,
        accuracy_standard,
        accuracy_minmax,
        accuracy_robust
    ]
})

print("\nFinal Model Comparison:")
print(results.to_string(index=False))


# ==================================================
# DAY 7 COMPLETED
# ==================================================

print("\nDay 7 completed successfully!")