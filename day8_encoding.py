# ==================================================
# DAY 8: ENCODING CATEGORICAL VARIABLES
# ==================================================

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    LabelEncoder
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ==================================================
# 1. CREATE SAMPLE MES DATASET
# ==================================================

data = {
    "station": [
        "Station_A", "Station_B", "Station_A",
        "Station_C", "Station_B", "Station_A",
        "Station_C", "Station_B", "Station_A",
        "Station_C", "Station_B", "Station_A"
    ],
    "product_model": [
        "Model_X", "Model_Y", "Model_Z",
        "Model_X", "Model_Y", "Model_Z",
        "Model_Y", "Model_X", "Model_X",
        "Model_Z", "Model_Z", "Model_Y"
    ],
    "machine_status": [
        "Low", "Medium", "High", "Low",
        "High", "Medium", "Low", "High",
        "Medium", "Low", "Medium", "High"
    ],
    "cycle_time": [
        30, 40, 50, 32, 45, 55,
        35, 48, 38, 60, 42, 58
    ],
    "overrun": [
        0, 0, 1, 0, 1, 1,
        0, 1, 0, 1, 0, 1
    ]
}

df = pd.DataFrame(data)

print("\nOriginal Dataset:")
print(df)


# ==================================================
# 2. ONE-HOT ENCODING
# ==================================================

print("\n========== ONE-HOT ENCODING ==========")

one_hot_df = pd.get_dummies(
    df,
    columns=["station"],
    dtype=int
)

print(one_hot_df.head())

# Example:
# Station_A -> station_Station_A = 1
#             station_Station_B = 0
#             station_Station_C = 0


# ==================================================
# 3. ORDINAL ENCODING
# ==================================================

print("\n========== ORDINAL ENCODING ==========")

ordinal_encoder = OrdinalEncoder(
    categories=[["Low", "Medium", "High"]]
)

df["machine_status_encoded"] = ordinal_encoder.fit_transform(
    df[["machine_status"]]
).ravel().astype(int)

print(
    df[[
        "machine_status",
        "machine_status_encoded"
    ]]
)


# ==================================================
# 4. LABEL ENCODING
# ==================================================

print("\n========== LABEL ENCODING ==========")

label_encoder = LabelEncoder()

df["product_model_encoded"] = label_encoder.fit_transform(
    df["product_model"]
)

print(
    df[[
        "product_model",
        "product_model_encoded"
    ]].drop_duplicates().sort_values("product_model")
)


# ==================================================
# 5. PREPARE FEATURES AND TARGET
# ==================================================

X = df[
    [
        "station",
        "product_model",
        "machine_status",
        "cycle_time"
    ]
]

y = df["overrun"]


# ==================================================
# 6. SPLIT DATA
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# ==================================================
# 7. ENCODE CATEGORICAL FEATURES SAFELY
# ==================================================

categorical_features = [
    "station",
    "product_model",
    "machine_status"
]

numerical_features = ["cycle_time"]

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


# ==================================================
# 8. TRAIN LOGISTIC REGRESSION MODEL
# ==================================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)

model.fit(X_train, y_train)


# ==================================================
# 9. PREDICT AND EVALUATE
# ==================================================

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n========== MODEL RESULTS ==========")
print("Actual values:", y_test.tolist())
print("Predictions:", predictions.tolist())
print("Accuracy:", round(accuracy, 4))


# ==================================================
# 10. PREDICT A NEW PRODUCTION RECORD
# ==================================================

new_record = pd.DataFrame({
    "station": ["Station_A"],
    "product_model": ["Model_X"],
    "machine_status": ["Medium"],
    "cycle_time": [44]
})

prediction = model.predict(new_record)

print("\nNew production record:")
print(new_record)

print(
    "Predicted overrun:",
    "Yes" if prediction[0] == 1 else "No"
)

print("\nDay 8 completed successfully!")