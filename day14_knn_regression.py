import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Historical data
X_train = np.array([
    [70, 5],
    [72, 6],
    [75, 8],
    [85, 20],
    [90, 25]
])

# Actual cycle times
y_train = np.array([60, 65, 70, 100, 110])

# Create KNN Regression model
model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsRegressor(
        n_neighbors=3,
        weights="uniform"
    ))
])

# Train model
model.fit(X_train, y_train)

# New machine operation
new_operation = np.array([[73, 7]])

# Predict cycle time
prediction = model.predict(new_operation)

print("Predicted cycle time:", prediction[0], "minutes")