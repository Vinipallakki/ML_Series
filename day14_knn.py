
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ==================================================
# 1. CREATE MANUFACTURING DATA
# ==================================================

np.random.seed(42)

n = 1000

df = pd.DataFrame({
    "temperature": np.random.normal(75, 10, n),
    "queue_length": np.random.randint(0, 30, n),
    "standard_cycle_time": np.random.uniform(50, 100, n),
    "station": np.random.choice(
        ["Station_A", "Station_B", "Station_C"], n
    )
})

# Station-specific effect on cycle time
station_effect = df["station"].map({
    "Station_A": 0,
    "Station_B": 8,
    "Station_C": 15
})

# Generate actual cycle time using input features
noise = np.random.normal(0, 8, n)

df["actual_cycle_time"] = (
    45
    + 0.5 * (df["temperature"] - 70)
    + 0.8 * df["queue_length"]
    + station_effect
    + noise
)

# Create target: 1 = Overrun, 0 = No Overrun
df["overrun"] = (
    df["actual_cycle_time"] >
    df["standard_cycle_time"]
).astype(int)

print("\n========== DATASET ==========")
print(df.head())

print("\nTarget distribution:")
print(df["overrun"].value_counts())

# ==================================================
# 2. SEPARATE X AND y
# ==================================================

X = df.drop(
    ["actual_cycle_time", "overrun"],
    axis=1
)

y = df["overrun"]

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
# 5. CREATE KNN MODEL
# ==================================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            KNeighborsClassifier(
                n_neighbors=5,
                weights="uniform",
                metric="euclidean"
            )
        )
    ]
)

# ==================================================
# 6. TRAIN MODEL
# ==================================================

model.fit(X_train, y_train)

print("\nKNN model training completed!")

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
# 8. EVALUATION
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
# 9. EXPERIMENT WITH DIFFERENT K VALUES
# ==================================================

print("\n========== K VALUE EXPERIMENT ==========")

for k in [1, 3, 5, 7, 9, 11, 15]:

    knn_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                KNeighborsClassifier(
                    n_neighbors=k
                )
            )
        ]
    )

    knn_pipeline.fit(X_train, y_train)

    predictions = knn_pipeline.predict(X_test)

    print(
        f"K={k} | "
        f"Accuracy={accuracy_score(y_test, predictions):.4f} | "
        f"Precision={precision_score(y_test, predictions):.4f} | "
        f"Recall={recall_score(y_test, predictions):.4f} | "
        f"F1={f1_score(y_test, predictions):.4f}"
    )

# ==================================================
# 10. PREDICT A NEW MACHINE OPERATION
# ==================================================

new_operation = pd.DataFrame({
    "temperature": [85],
    "queue_length": [20],
    "standard_cycle_time": [70],
    "station": ["Station_B"]
})

new_prediction = model.predict(new_operation)[0]

new_probability = model.predict_proba(
    new_operation
)[0]

print("\n========== NEW OPERATION PREDICTION ==========")

print("No Overrun Probability:", round(new_probability[0], 4))
print("Overrun Probability:", round(new_probability[1], 4))

print("Predicted Class:", new_prediction)

if new_prediction == 1:
    print("Result: Machine operation may exceed standard time.")
else:
    print("Result: Machine operation is expected within standard time.")

# ==================================================
# 11. FIND NEAREST NEIGHBOURS
# ==================================================

preprocessor_fitted = model.named_steps["preprocessor"]
knn_model = model.named_steps["classifier"]

new_operation_scaled = preprocessor_fitted.transform(
    new_operation
)

distances, indices = knn_model.kneighbors(
    new_operation_scaled
)

nearest_records = X_train.iloc[indices[0]].copy()
nearest_records["actual_overrun"] = y_train.iloc[indices[0]].values
nearest_records["distance"] = distances[0]

print("\n========== NEAREST 5 HISTORICAL OPERATIONS ==========")
print(nearest_records)
