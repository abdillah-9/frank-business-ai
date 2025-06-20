import pandas as pd
import numpy as np
import random
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Generate synthetic childcare data
def generate_user_childcare():
    data = []
    for _ in range(300):  # simulate 300 users
        baby_age_month = random.randint(0, 24)
        gender = random.choice([0, 1])  # 0 = male, 1 = female
        birth_weight = round(random.uniform(2.0, 4.5), 2)
        current_weight = round(birth_weight + random.uniform(-0.5, 3.0), 2)
        feeding_type = random.choice([0, 1, 2])  # 0 = breastfeeding, 1 = formula, 2 = mixed
        feeding_frequency = random.randint(5, 12)
        sleep_hours = random.randint(10, 18)

        # Determine condition_ok
        condition_ok = 1
        if current_weight < birth_weight or feeding_frequency < 8 or sleep_hours < 14:
            condition_ok = 0

        data.append({
            "baby_age_month": baby_age_month,
            "gender": gender,
            "birth_weight": birth_weight,
            "current_weight": current_weight,
            "feeding_type": feeding_type,
            "feeding_frequency": feeding_frequency,
            "sleep_hours": sleep_hours,
            "condition_ok": condition_ok
        })

    return pd.DataFrame(data)

# Generate and prepare data
df = generate_user_childcare()

X = df[[
    "baby_age_month", "gender", "birth_weight", "current_weight",
    "feeding_type", "feeding_frequency", "sleep_hours"
]]
y = df["condition_ok"]

# Define and train pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(random_state=42))
])

pipeline.fit(X, y)

# Save model
joblib.dump(pipeline, "childcareModel.joblib")
print("✅ Childcare model trained and saved as 'childcareModel.joblib'")
