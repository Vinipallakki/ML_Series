import numpy as np


# =========================================================
# 1. CREATE A 1D ARRAY
# =========================================================

x = np.array([10, 20, 30, 40, 50])

print("Array:", x)

print("Number of dimensions:", x.ndim)

print("Shape:", x.shape)

print("Size:", x.size)


# =========================================================
# 2. INDEXING
# =========================================================

print("\n--- Indexing ---")

print("First element:", x[0])

print("Second element:", x[1])

print("Last element:", x[-1])


# =========================================================
# 3. SLICING
# =========================================================

print("\n--- Slicing ---")

print("First 3 elements:", x[0:3])

print("Elements from index 2:", x[2:])

print("Every second element:", x[::2])


# =========================================================
# 4. MATHEMATICAL OPERATIONS
# =========================================================

print("\n--- Mathematical Operations ---")

print("Original:", x)

print("Add 10:", x + 10)

print("Subtract 5:", x - 5)

print("Multiply by 2:", x * 2)

print("Divide by 10:", x / 10)


# =========================================================
# 5. STATISTICS
# =========================================================

print("\n--- Statistics ---")

print("Mean:", np.mean(x))

print("Maximum:", np.max(x))

print("Minimum:", np.min(x))

print("Sum:", np.sum(x))

print("Standard deviation:", np.std(x))


# =========================================================
# 6. CREATE A 2D ARRAY
# =========================================================

X = np.array([
    [1, 50],
    [2, 55],
    [3, 60],
    [4, 65]
])

print("\n--- 2D Array ---")

print(X)

print("Number of dimensions:", X.ndim)

print("Shape:", X.shape)

print("Size:", X.size)


# =========================================================
# 7. 2D ARRAY INDEXING
# =========================================================

print("\n--- 2D Indexing ---")

print("First row:", X[0])

print("Second row:", X[1])

print("First value:", X[0, 0])

print("First row, second column:", X[0, 1])

print("Second row, first column:", X[1, 0])


# =========================================================
# 8. SELECT COLUMNS
# =========================================================

print("\n--- Column Selection ---")

print("First column:")
print(X[:, 0])

print("Second column:")
print(X[:, 1])


# =========================================================
# 9. SELECT ROWS
# =========================================================

print("\n--- Row Selection ---")

print("First two rows:")
print(X[0:2])

print("Last two rows:")
print(X[2:4])


# =========================================================
# 10. VECTORIZATION
# =========================================================

print("\n--- Vectorization ---")

numbers = np.array([10, 20, 30, 40])

result = numbers * 2

print("Original:", numbers)

print("After multiplying by 2:", result)


# =========================================================
# 11. STUDENT EXAMPLE
# =========================================================

print("\n--- Student Example ---")

hours = np.array([2, 4, 6, 8])

attendance = np.array([60, 70, 80, 90])

print("Hours:", hours)

print("Attendance:", attendance)


# Calculate a simple score

score = hours * 0.6 + attendance * 0.4

print("Score:", score)


# =========================================================
# 12. ML DATASET EXAMPLE
# =========================================================

print("\n--- ML Dataset ---")

X = np.array([
    [2, 60],
    [4, 70],
    [6, 80],
    [8, 90]
])

y = np.array([0, 0, 1, 1])

print("Features X:")
print(X)

print("Target y:")
print(y)

print("X shape:", X.shape)

print("y shape:", y.shape)