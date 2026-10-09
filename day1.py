import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# -----------------------------
# 1. Create sample data
# -----------------------------

data = {
    "hours_studied": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "attendance": [50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
    "passed": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

print(df)


# -----------------------------
# 2. Features and target
# -----------------------------

X = df[["hours_studied", "attendance"]]

y = df["passed"]


# -----------------------------
# 3. Train/Test split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -----------------------------
# 4. Create model
# -----------------------------

model = LogisticRegression()


# -----------------------------
# 5. Train
# -----------------------------

model.fit(X_train, y_train)


# -----------------------------
# 6. Predict
# -----------------------------

predictions = model.predict(X_test)

print("Predictions:", predictions)
print("Actual:", y_test.values)


# -----------------------------
# 7. Evaluate
# -----------------------------

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)


# -----------------------------
# 8. Predict new student
# -----------------------------

new_student = [[7, 15]]

prediction = model.predict(new_student)

probability = model.predict_proba(new_student)

print("Prediction:", prediction)
print("Probability:", probability)