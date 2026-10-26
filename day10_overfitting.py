
# import numpy as np
# import pandas as pd

# from sklearn.model_selection import train_test_split
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.pipeline import Pipeline
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score, f1_score

# # 1. Generate sample MES data
# np.random.seed(42)
# n = 1000

# df = pd.DataFrame({
#     "station": np.random.choice(
#         ["Station_A", "Station_B", "Station_C"], n
#     ),
#     "temperature": np.random.normal(75, 10, n),
#     "standard_cycle_time": np.random.uniform(30, 120, n),
#     "queue_length": np.random.randint(0, 30, n)
# })

# # 2. Generate a synthetic overrun target
# logit = (
#     -4.0
#     + 0.035 * (df["temperature"] - 70)
#     + 0.08 * df["queue_length"]
#     + 0.025 * (df["standard_cycle_time"] - 60)
#     + df["station"].map({
#         "Station_A": 0.0,
#         "Station_B": 0.4,
#         "Station_C": 0.8
#     })
# )

# probability = 1 / (1 + np.exp(-logit))
# df["overrun"] = np.random.binomial(1, probability)

# # 3. Separate features and target
# X = df[
#     ["station", "temperature",
#      "standard_cycle_time", "queue_length"]
# ]
# y = df["overrun"]

# # 4. Split into training and testing data
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y,
#     test_size=0.20,
#     random_state=42,
#     stratify=y
# )

# # 5. Encode the categorical station column
# preprocessor = ColumnTransformer([
#     (
#         "categorical",
#         OneHotEncoder(handle_unknown="ignore"),
#         ["station"]
#     ),
#     (
#         "numerical",
#         "passthrough",
#         ["temperature", "standard_cycle_time", "queue_length"]
#     )
# ])

# # 6. Compare different model complexities
# depths = [1, 3, 5, None]

# for depth in depths:
#     model = Pipeline([
#         ("preprocessor", preprocessor),
#         ("classifier", DecisionTreeClassifier(
#             max_depth=depth,
#             min_samples_leaf=5,
#             random_state=42
#         ))
#     ])

#     # Learn from training records
#     model.fit(X_train, y_train)

#     # Evaluate on both datasets
#     train_pred = model.predict(X_train)
#     test_pred = model.predict(X_test)

#     train_accuracy = accuracy_score(y_train, train_pred)
#     test_accuracy = accuracy_score(y_test, test_pred)

#     train_f1 = f1_score(y_train, train_pred, zero_division=0)
#     test_f1 = f1_score(y_test, test_pred, zero_division=0)

#     print(f"\nMax depth: {depth}")
#     print(f"Training accuracy: {train_accuracy:.4f}")
#     print(f"Testing accuracy:  {test_accuracy:.4f}")
#     print(f"Training F1:       {train_f1:.4f}")
#     print(f"Testing F1:        {test_f1:.4f}")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. Generate manufacturing data
np.random.seed(42)

n = 1000

df = pd.DataFrame({
    "temperature": np.random.normal(75, 10, n),
    "standard_cycle_time": np.random.uniform(30, 120, n),
    "queue_length": np.random.randint(0, 30, n),
    "station": np.random.choice(
        [0, 1, 2], n
    )
})

# 2. Generate target: 1 = overrun, 0 = no overrun
score = (
    -2.0
    + 0.04 * (df["temperature"] - 70)
    + 0.08 * df["queue_length"]
    + 0.015 * (df["standard_cycle_time"] - 60)
    + 0.4 * df["station"]
)

probability = 1 / (1 + np.exp(-score))

df["overrun"] = np.random.binomial(1, probability)

# 3. Separate features and target
X = df[
    [
        "temperature",
        "standard_cycle_time",
        "queue_length",
        "station"
    ]
]

y = df["overrun"]

# 4. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 5. Create three models with different complexity
models = {
    "Underfitting": DecisionTreeClassifier(
        max_depth=1,
        random_state=42
    ),

    "Controlled Tree": DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=5,
        random_state=42
    ),

    "Overfitting Risk": DecisionTreeClassifier(
        max_depth=None,
        min_samples_leaf=1,
        random_state=42
    )
}

# 6. Train and evaluate each model
results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_accuracy = accuracy_score(
        y_train, train_predictions
    )

    test_accuracy = accuracy_score(
        y_test, test_predictions
    )

    results.append({
        "Model": name,
        "Training Accuracy": train_accuracy,
        "Testing Accuracy": test_accuracy
    })

    print(f"\n========== {name} ==========")
    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Testing accuracy:  {test_accuracy:.4f}")

# 7. Compare the results
results_df = pd.DataFrame(results)

print("\n========== FINAL COMPARISON ==========")
print(results_df.round(4).to_string(index=False))

# 8. Visualize training vs testing accuracy
results_df.set_index("Model")[
    ["Training Accuracy", "Testing Accuracy"]
].plot(kind="bar", ylim=(0, 1.05))

plt.title("Underfitting vs Controlled Fit vs Overfitting")
plt.ylabel("Accuracy")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
