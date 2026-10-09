import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =====================================================
# 1. CREATE DATA
# =====================================================

data = {
    "name": [
        "Rahul", "Priya", "Amit", "Sneha", "Kiran",
        "Arun", "Divya", "Ravi", "Anil", "Meena"
    ],

    "age": [
        25, 30, 28, 35, 40,
        27, 32, 45, 29, 38
    ],

    "salary": [
        50000, 70000, 60000, 90000, 100000,
        55000, 75000, 150000, 65000, 85000
    ],

    "experience": [
        2, 5, 3, 8, 12,
        3, 6, 18, 4, 10
    ],

    "department": [
        "IT", "HR", "IT", "Finance", "IT",
        "IT", "HR", "IT", "Finance", "IT"
    ]
}


df = pd.DataFrame(data)


# =====================================================
# 2. BASIC DATA CHECK
# =====================================================

print("\n========== DATA ==========")

print(df)


print("\n========== SHAPE ==========")

print(df.shape)


print("\n========== INFO ==========")

df.info()


print("\n========== DESCRIBE ==========")

print(df.describe())


# =====================================================
# 3. MISSING VALUES
# =====================================================

print("\n========== MISSING VALUES ==========")

print(df.isnull().sum())


# =====================================================
# 4. DUPLICATES
# =====================================================

print("\n========== DUPLICATES ==========")

print(df.duplicated().sum())


# =====================================================
# 5. MEAN
# =====================================================

print("\n========== MEAN ==========")

print("Average salary:", df["salary"].mean())

print("Average age:", df["age"].mean())

print("Average experience:", df["experience"].mean())


# =====================================================
# 6. MEDIAN
# =====================================================

print("\n========== MEDIAN ==========")

print("Median salary:", df["salary"].median())

print("Median age:", df["age"].median())

print("Median experience:", df["experience"].median())


# =====================================================
# 7. MODE
# =====================================================

print("\n========== MODE ==========")

print("Department mode:")

print(df["department"].mode())


# =====================================================
# 8. STANDARD DEVIATION
# =====================================================

print("\n========== STANDARD DEVIATION ==========")

print("Salary std:", df["salary"].std())

print("Age std:", df["age"].std())

print("Experience std:", df["experience"].std())


# =====================================================
# 9. VARIANCE
# =====================================================

print("\n========== VARIANCE ==========")

print("Salary variance:", df["salary"].var())


# =====================================================
# 10. QUARTILES
# =====================================================

print("\n========== QUARTILES ==========")

Q1 = df["salary"].quantile(0.25)

Q2 = df["salary"].quantile(0.50)

Q3 = df["salary"].quantile(0.75)

print("Q1:", Q1)

print("Q2 / Median:", Q2)

print("Q3:", Q3)


# =====================================================
# 11. IQR
# =====================================================

print("\n========== IQR ==========")

IQR = Q3 - Q1

print("IQR:", IQR)


# =====================================================
# 12. OUTLIER BOUNDS
# =====================================================

lower_bound = Q1 - 1.5 * IQR

upper_bound = Q3 + 1.5 * IQR

print("Lower bound:", lower_bound)

print("Upper bound:", upper_bound)


# =====================================================
# 13. FIND OUTLIERS
# =====================================================

print("\n========== OUTLIERS ==========")

outliers = df[
    (df["salary"] < lower_bound) |
    (df["salary"] > upper_bound)
]

print(outliers)


# =====================================================
# 14. CORRELATION
# =====================================================

print("\n========== CORRELATION ==========")

correlation = df[
    ["age", "salary", "experience"]
].corr()

print(correlation)


# =====================================================
# 15. GROUPBY
# =====================================================

print("\n========== DEPARTMENT SALARY ==========")

department_salary = (
    df.groupby("department")["salary"]
    .mean()
)

print(department_salary)


# =====================================================
# 16. HISTOGRAM
# =====================================================

plt.hist(df["salary"])

plt.xlabel("Salary")

plt.ylabel("Number of Employees")

plt.title("Salary Distribution")

plt.show()


# =====================================================
# 17. BOX PLOT
# =====================================================

plt.boxplot(df["salary"])

plt.ylabel("Salary")

plt.title("Salary Box Plot")

plt.show()


# =====================================================
# 18. SCATTER PLOT
# =====================================================

plt.scatter(
    df["experience"],
    df["salary"]
)

plt.xlabel("Experience")

plt.ylabel("Salary")

plt.title("Experience vs Salary")

plt.show()


# =====================================================
# 19. AGE VS SALARY
# =====================================================

plt.scatter(
    df["age"],
    df["salary"]
)

plt.xlabel("Age")

plt.ylabel("Salary")

plt.title("Age vs Salary")

plt.show()


# =====================================================
# 20. FINAL SUMMARY
# =====================================================

print("\n========== EDA SUMMARY ==========")

print("Rows:", df.shape[0])

print("Columns:", df.shape[1])

print("Average salary:", df["salary"].mean())

print("Median salary:", df["salary"].median())

print("Salary standard deviation:", df["salary"].std())

print("Number of missing values:", df.isnull().sum().sum())

print("Number of duplicate rows:", df.duplicated().sum())

print("\nCorrelation:")

print(correlation)