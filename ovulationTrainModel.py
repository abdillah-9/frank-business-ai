import pandas as pd
import numpy as np
import random

STRESS_LEVEL_MAP = {"low": 0, "medium": 1, "high": 2}

def random_stress():
    return random.choice(list(STRESS_LEVEL_MAP.keys()))

# Generate many cycles per user, then aggregate per user
users_data = []

for user_id in range(200):
    cycles = []
    for _ in range(random.randint(3, 10)):  # simulate multiple cycles per user
        age = random.randint(15, 45)
        cycle_length_days = random.randint(24, 35)
        period_duration_days = random.randint(3, 7)
        is_cycle_regular = random.choice([True, False])
        stress_level = random_stress()
        sleep_hours = random.randint(4, 10)
        day_week_exercise = random.randint(0, 7)

        cycles.append({
            "age": age,
            "cycle_length_days": cycle_length_days,
            "period_duration_days": period_duration_days,
            "is_cycle_regular": int(is_cycle_regular),
            "stress_level_encoded": STRESS_LEVEL_MAP[stress_level],
            "sleep_hours": sleep_hours,
            "day_week_exercise": day_week_exercise
        })

    df_cycles = pd.DataFrame(cycles)
    # Compute aggregated features per user
    age = df_cycles["age"].iloc[0]  # age usually fixed per user
    regular_ratio = df_cycles["is_cycle_regular"].mean()
    avg_stress = df_cycles["stress_level_encoded"].mean()
    avg_sleep = df_cycles["sleep_hours"].mean()
    avg_exercise = df_cycles["day_week_exercise"].mean()
    cycle_std = df_cycles["cycle_length_days"].std()

    # Compute target for user (simulate)
    condition_ok = 1 if (regular_ratio > 0.6 and avg_sleep >= 6 and avg_stress < 1.5) else 0

    users_data.append({
        "age": age,
        "regular_ratio": regular_ratio,
        "avg_stress": avg_stress,
        "avg_sleep": avg_sleep,
        "avg_exercise": avg_exercise,
        "cycle_std": 0 if np.isnan(cycle_std) else cycle_std,
        "condition_ok": condition_ok
    })

df_users = pd.DataFrame(users_data)

X = df_users[["age", "regular_ratio", "avg_stress", "avg_sleep", "avg_exercise", "cycle_std"]]
y = df_users["condition_ok"]

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(random_state=42))
])

pipeline.fit(X, y)
joblib.dump(pipeline, "ovulationFaridaModel.joblib")

print("✅ Model trained on aggregated user features and saved")
