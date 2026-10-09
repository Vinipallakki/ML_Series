import pandas as pd
import numpy as np


# =====================================================
# 1. CREATE RAW DATA
# =====================================================

df = pd.DataFrame({
    "serial": [
        "S001",
        "S002",
        "S003",
        "S004",
        "S005"
    ],

    "station": [
        "S01",
        "S01",
        "S02",
        "S02",
        "S01"
    ],

    "model": [
        "M100",
        "M100",
        "M200",
        "M200",
        "M100"
    ],

    "standard_cycle_sec": [
        400,
        400,
        500,
        500,
        400
    ],

    "actual_duration_sec": [
        450,
        600,
        480,
        700,
        420
    ],

    "andon_count": [
        0,
        2,
        0,
        3,
        1
    ],

    "timestamp": [
        "2026-10-01 08:30",
        "2026-10-01 10:15",
        "2026-10-01 14:20",
        "2026-10-01 18:45",
        "2026-10-01 20:10"
    ]
})


print("\n========== RAW DATA ==========")

print(df)


# =====================================================
# 2. CONVERT TIMESTAMP
# =====================================================

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)


# =====================================================
# 3. TIME FEATURES
# =====================================================

df["hour"] = df["timestamp"].dt.hour

df["day"] = df["timestamp"].dt.day

df["month"] = df["timestamp"].dt.month

df["weekday"] = df["timestamp"].dt.dayofweek


# =====================================================
# 4. WEEKEND FEATURE
# =====================================================

df["is_weekend"] = (
    df["weekday"] >= 5
).astype(int)


# =====================================================
# 5. SHIFT FEATURE
# =====================================================

def get_shift(hour):

    if 6 <= hour < 14:
        return "Morning"

    elif 14 <= hour < 22:
        return "Afternoon"

    else:
        return "Night"


df["shift"] = df["hour"].apply(
    get_shift
)


# =====================================================
# 6. DURATION RATIO
# =====================================================

df["duration_ratio"] = (
    df["actual_duration_sec"]
    / df["standard_cycle_sec"]
)


# =====================================================
# 7. OVERRUN FEATURE
# =====================================================

df["overrun"] = (
    df["duration_ratio"] > 1.2
).astype(int)


# =====================================================
# 8. STATION AVERAGE
# =====================================================

df["station_avg_duration"] = (
    df.groupby("station")["actual_duration_sec"]
    .transform("mean")
)


# =====================================================
# 9. MODEL AVERAGE
# =====================================================

df["model_avg_duration"] = (
    df.groupby("model")["actual_duration_sec"]
    .transform("mean")
)


# =====================================================
# 10. STATION OVERRUN RATE
# =====================================================

df["station_overrun_rate"] = (
    df.groupby("station")["overrun"]
    .transform("mean")
)


# =====================================================
# 11. MODEL OVERRUN RATE
# =====================================================

df["model_overrun_rate"] = (
    df.groupby("model")["overrun"]
    .transform("mean")
)


# =====================================================
# 12. CURRENT VS STATION AVERAGE
# =====================================================

df["current_vs_station_avg"] = (
    df["actual_duration_sec"]
    / df["station_avg_duration"]
)


# =====================================================
# 13. LOG TRANSFORMATION
# =====================================================

df["log_duration"] = np.log1p(
    df["actual_duration_sec"]
)


# =====================================================
# 14. DISPLAY FEATURE DATASET
# =====================================================

print("\n========== FEATURE DATASET ==========")

print(df)


# =====================================================
# 15. SHOW IMPORTANT FEATURES
# =====================================================

features = [
    "standard_cycle_sec",
    "actual_duration_sec",
    "andon_count",
    "hour",
    "weekday",
    "is_weekend",
    "duration_ratio",
    "station_avg_duration",
    "model_avg_duration",
    "station_overrun_rate",
    "model_overrun_rate",
    "current_vs_station_avg"
]


print("\n========== ML FEATURES ==========")

print(df[features])


# =====================================================
# 16. CREATE X AND y
# =====================================================

X = df[
    features
]

y = df[
    "overrun"
]


print("\n========== X ==========")

print(X)


print("\n========== y ==========")

print(y)


# =====================================================
# 17. SHAPES
# =====================================================

print("\n========== SHAPES ==========")

print("X shape:", X.shape)

print("y shape:", y.shape)