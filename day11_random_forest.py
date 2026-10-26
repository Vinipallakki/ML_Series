
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# 1. Generate sample MES data
np.random.seed(42)
n = 1000

df = pd.DataFrame({
    "station": np.random.choice(
        ["Station_A", "Station_B", "Station_C"], n
    ),
    "temperature": np.random.normal(75, 10, n),
    "standard_cycle_time": np.random.uniform(30, 120, n),
    "queue_length": np.random.randint(0, 30, n)
})

# 2. Generate synthetic overrun labels
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

# 3. Separate features and target
X = df[
    ["station", "temperature",
     "standard_cycle_time", "queue_length"]
]
y = df["overrun"]

# 4. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 5. Preprocess the categorical feature
preprocessor = ColumnTransformer([
    (
        "categorical",
        OneHotEncoder(handle_unknown="ignore"),
        ["station"]
    ),
    (
        "numerical",
        "passthrough",
        ["temperature", "standard_cycle_time", "queue_length"]
    )
])

# 6. Define two models
models = {
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=5,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
}

# 7. Train and evaluate both models
for name, classifier in models.items():

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    print(f"\n========== {name} ==========")
    print(
        "Training accuracy:",
        round(accuracy_score(y_train, train_pred), 4)
    )
    print(
        "Testing accuracy:",
        round(accuracy_score(y_test, test_pred), 4)
    )
    print(
        "Test precision:",
        round(precision_score(
            y_test, test_pred, zero_division=0
        ), 4)
    )
    print(
        "Test recall:",
        round(recall_score(
            y_test, test_pred, zero_division=0
        ), 4)
    )
    print(
        "Test F1:",
        round(f1_score(
            y_test, test_pred, zero_division=0
        ), 4)
    )

    print("Confusion matrix:")
    print(confusion_matrix(y_test, test_pred))
