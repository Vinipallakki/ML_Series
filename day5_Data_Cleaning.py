import pandas as pd
import numpy as np


# =====================================================
# 1. CREATE DIRTY DATA
# =====================================================

data = {
    "name": [
        " Rahul ",
        "Priya",
        "Amit",
        "Sneha",
        "Kiran",
        "Rahul"
    ],

    "age": [
        25,
        30,
        np.nan,
        35,
        150,
        25
    ],

    "salary": [
        50000,
        70000,
        60000,
        np.nan,
        100000,
        50000
    ],

    "experience": [
        2,
        5,
        3,
        8,
        np.nan,
        2
    ],

    "department": [
        " IT ",
        "HR",
        "IT",
        "Finance",
        "IT",
        " IT "
    ]
}


df = pd.DataFrame(data)


print("\n========== ORIGINAL DATA ==========")

print(df)


# =====================================================
# 2. BASIC INFORMATION
# =====================================================

print("\n========== INFO ==========")

df.info()


print("\n========== DATA TYPES ==========")

print(df.dtypes)


# =====================================================
# 3. CHECK MISSING VALUES
# =====================================================

print("\n========== MISSING VALUES ==========")

print(df.isnull().sum())


print("\n========== MISSING PERCENTAGE ==========")

missing_percentage = (
    df.isnull().sum() / len(df)
) * 100

print(missing_percentage)


# =====================================================
# 4. REMOVE DUPLICATES
# =====================================================

print("\n========== DUPLICATES ==========")

print("Duplicate rows:",
      df.duplicated().sum())


df = df.drop_duplicates()


# =====================================================
# 5. CLEAN STRING VALUES
# =====================================================

df["name"] = (
    df["name"]
    .str.strip()
)


df["department"] = (
    df["department"]
    .str.strip()
    .str.upper()
)


print("\n========== CLEANED STRINGS ==========")

print(df)


# =====================================================
# 6. CONVERT DATA TYPES
# =====================================================

df["age"] = pd.to_numeric(
    df["age"],
    errors="coerce"
)


df["salary"] = pd.to_numeric(
    df["salary"],
    errors="coerce"
)


df["experience"] = pd.to_numeric(
    df["experience"],
    errors="coerce"
)


print("\n========== AFTER TYPE CONVERSION ==========")

print(df.dtypes)


# =====================================================
# 7. CHECK INVALID AGE
# =====================================================

print("\n========== INVALID AGE ==========")

invalid_age = df[
    (df["age"] < 0) |
    (df["age"] > 100)
]

print(invalid_age)


# =====================================================
# 8. HANDLE INVALID AGE
# =====================================================

df.loc[
    (df["age"] < 0) |
    (df["age"] > 100),
    "age"
] = np.nan


# =====================================================
# 9. CHECK INVALID SALARY
# =====================================================

print("\n========== INVALID SALARY ==========")

invalid_salary = df[
    df["salary"] < 0
]

print(invalid_salary)


# =====================================================
# 10. FILL NUMERICAL MISSING VALUES
# =====================================================

df["age"] = df["age"].fillna(
    df["age"].median()
)


df["salary"] = df["salary"].fillna(
    df["salary"].median()
)


df["experience"] = df["experience"].fillna(
    df["experience"].median()
)


# =====================================================
# 11. CHECK MISSING VALUES AGAIN
# =====================================================

print("\n========== MISSING VALUES AFTER CLEANING ==========")

print(df.isnull().sum())


# =====================================================
# 12. CHECK CATEGORIES
# =====================================================

print("\n========== DEPARTMENTS ==========")

print(df["department"].unique())


# =====================================================
# 13. VALIDATE DEPARTMENTS
# =====================================================

valid_departments = [
    "IT",
    "HR",
    "FINANCE"
]


invalid_departments = df[
    ~df["department"].isin(valid_departments)
]


print("\n========== INVALID DEPARTMENTS ==========")

print(invalid_departments)


# =====================================================
# 14. OUTLIER DETECTION USING IQR
# =====================================================

Q1 = df["salary"].quantile(0.25)

Q3 = df["salary"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR

upper_bound = Q3 + 1.5 * IQR


print("\n========== IQR ==========")

print("Q1:", Q1)

print("Q3:", Q3)

print("IQR:", IQR)

print("Lower bound:", lower_bound)

print("Upper bound:", upper_bound)


# =====================================================
# 15. FIND SALARY OUTLIERS
# =====================================================

outliers = df[
    (df["salary"] < lower_bound) |
    (df["salary"] > upper_bound)
]


print("\n========== SALARY OUTLIERS ==========")

print(outliers)


# =====================================================
# 16. FINAL CLEAN DATA
# =====================================================

print("\n========== FINAL CLEAN DATA ==========")

print(df)


# =====================================================
# 17. FINAL VALIDATION
# =====================================================

print("\n========== FINAL VALIDATION ==========")

print("Rows:", df.shape[0])

print("Columns:", df.shape[1])

print("Missing values:",
      df.isnull().sum().sum())

print("Duplicate rows:",
      df.duplicated().sum())

print("\nData types:")

print(df.dtypes)