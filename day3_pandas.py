import pandas as pd
import numpy as np


# =====================================================
# 1. CREATE DATA
# =====================================================

data = {
    "name": ["Rahul", "Priya", "Amit", "Sneha", "Kiran"],
    "age": [25, 30, 28, 35, 40],
    "saguivlary": [50000, 70000, 60000, 90000, 100000],
    "experience": [2, 5, 3, 8, 12],
    "department": ["IT", "HR", "IT", "Finance", "IT"]
}


# =====================================================
# 2. CREATE DATAFRAME
# =====================================================

df = pd.DataFrame(data)

print("\n--- DATA ---")
print(df)


# =====================================================
# 3. BASIC INFORMATION
# =====================================================

print("\n--- HEAD ---")
print(df.head())

print("\n--- TAIL ---")
print(df.tail())

print("\n--- SHAPE ---")
print(df.shape)

print("\n--- COLUMNS ---")
print(df.columns)

print("\n--- INFO ---")
df.info()

print("\n--- DESCRIBE ---")
print(df.describe())


# =====================================================
# 4. SELECT ONE COLUMN
# =====================================================

print("\n--- SALARY COLUMN ---")

salary = df["saguivlary"]

print(salary)


# =====================================================
# 5. SELECT MULTIPLE COLUMNS
# =====================================================

print("\n--- MULTIPLE COLUMNS ---")

X = df[["age", "saguivlary", "experience"]]

print(X)


# =====================================================
# 6. CREATE TARGET
# =====================================================

df["promoted"] = [0, 1, 0, 1, 1]

y = df["promoted"]

print("\n--- TARGET ---")
print(y)


# =====================================================
# 7. FILTERING
# =====================================================

print("\n--- IT EMPLOYEES ---")

it_employees = df[
    df["department"] == "IT"
]

print(it_employees)


print("\n--- SALARY > 70000 ---")

high_salary = df[
    df["saguivlary"] > 70000
]

print(high_salary)


print("\n--- IT AND SALARY > 60000 ---")

result = df[
    (df["department"] == "IT") &
    (df["saguivlary"] > 60000)
]

print(result)


# =====================================================
# 8. ADD NEW COLUMN
# =====================================================

df["salary_lakh"] = (
    df["saguivlary"] / 100000
)

print("\n--- SALARY IN LAKHS ---")

print(df[["name", "saguivlary", "salary_lakh"]])


# =====================================================
# 9. FEATURE ENGINEERING
# =====================================================

df["salary_per_experience"] = (
    df["saguivlary"] / df["experience"]
)

print("\n--- FEATURE ENGINEERING ---")

print(
    df[
        [
            "name",
            "saguivlary",
            "experience",
            "salary_per_experience"
        ]
    ]
)


# =====================================================
# 10. SORTING
# =====================================================

print("\n--- SORT BY SALARY ---")

sorted_df = df.sort_values(
    "saguivlary",
    ascending=False
)

print(sorted_df)


# =====================================================
# 11. GROUPBY
# =====================================================

print("\n--- AVERAGE SALARY BY DEPARTMENT ---")

avg_salary = (
    df.groupby("department")["saguivlary"]
    .mean()
)

print(avg_salary)


# =====================================================
# 12. GROUPBY WITH MULTIPLE STATISTICS
# =====================================================

print("\n--- DEPARTMENT STATISTICS ---")

stats = (
    df.groupby("department")["saguivlary"]
    .agg(["mean", "min", "max", "count"])
)

print(stats)


# =====================================================
# 13. MISSING VALUES
# =====================================================

print("\n--- MISSING VALUE EXAMPLE ---")

missing_data = {
    "age": [25, 30, np.nan, 35, 40],
    "saguivlary": [50000, 70000, 60000, np.nan, 100000]
}

missing_df = pd.DataFrame(missing_data)

print(missing_df)


# =====================================================
# 14. CHECK MISSING VALUES
# =====================================================

print("\n--- MISSING VALUE COUNT ---")

print(missing_df.isnull().sum())


# =====================================================
# 15. FILL MISSING VALUES
# =====================================================

missing_df["age"] = missing_df["age"].fillna(
    missing_df["age"].mean()
)

missing_df["saguivlary"] = missing_df["saguivlary"].fillna(
    missing_df["saguivlary"].mean()
)

print("\n--- AFTER FILLING MISSING VALUES ---")

print(missing_df)


# =====================================================
# 16. DROP DUPLICATES
# =====================================================

df = df.drop_duplicates()

print("\n--- AFTER REMOVING DUPLICATES ---")

print(df)


# =====================================================
# 17. LOC
# =====================================================

print("\n--- LOC ---")

print(df.loc[0])

print("\nSalary of first employee:")

print(df.loc[0, "saguivlary"])


# =====================================================
# 18. ILOC
# =====================================================

print("\n--- ILOC ---")

print(df.iloc[0])

print("\nFirst row, second column:")

print(df.iloc[0, 1])


# =====================================================
# 19. MERGE TWO DATAFRAMES
# =====================================================

employees = pd.DataFrame({
    "employee_id": [1, 2, 3],
    "name": ["Rahul", "Priya", "Amit"]
})

salary = pd.DataFrame({
    "employee_id": [1, 2, 3],
    "saguivlary": [50000, 70000, 60000]
})

merged = pd.merge(
    employees,
    salary,
    on="employee_id",
    how="inner"
)

print("\n--- MERGED DATA ---")

print(merged)


# =====================================================
# 20. FINAL ML DATA
# =====================================================

print("\n--- FINAL X AND y ---")

X = df[
    [
        "age",
        "saguivlary",
        "experience"
    ]
]

y = df["promoted"]

print("Features X:")
print(X)

print("\nTarget y:")
print(y)

print("\nX shape:")
print(X.shape)

print("\ny shape:")
print(y.shape)