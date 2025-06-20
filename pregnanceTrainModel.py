import pandas as pd
import numpy as np
import random
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Generate synthetic pregnancy data
def generate_user_pregnancy():
    data = []
    for _ in range(300):  # simulate 300 users
        pregnance_week = random.randint(1, 40)
        featus_number = random.choice([1, 1, 1, 2])  # most common is 1
        is_smoking = random.choice([0, 1])
        is_drinking = random.choice([0, 1])
        mental_health_problem = random.choice([0, 1])
        fetal_HR = random.randint(110, 160) if random.random() > 0.1 else 0  # simulate 0 if missing
        mother_HR = random.randint(60, 100) if random.random() > 0.1 else 0

        # Simulate condition_ok
        condition_ok = 1
        if is_smoking or is_drinking or mental_health_problem or fetal_HR < 100 or mother_HR < 50:
            condition_ok = 0

        data.append({
            "pregnance_week": pregnance_week,
            "featus_number": featus_number,
            "is_smoking": is_smoking,
            "is_drinking": is_drinking,
            "mental_health_problem": mental_health_problem,
            "fetal_HR": fetal_HR,
            "mother_HR": mother_HR,
            "condition_ok": condition_ok
        })

    return pd.DataFrame(data)

# Generate and prepare data
df = generate_user_pregnancy()

X = df[[
        "pregnance_week", "featus_number", "is_smoking", 
        "is_drinking", "mental_health_problem", "fetal_HR", "mother_HR"
    ]]
y = df["condition_ok"]

# Define and train pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(random_state=42))
])

pipeline.fit(X, y)

# Save model
joblib.dump(pipeline, "pregnancyFaridaModel.joblib")
print("✅ Pregnancy model trained and saved as 'pregnancyFaridaModel.joblib'")
