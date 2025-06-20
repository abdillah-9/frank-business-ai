# train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Dummy training data
data = pd.DataFrame([
    {"budget": 100, "spent": 140, "over_budget": 1},
    {"budget": 200, "spent": 180, "over_budget": 0},
    {"budget": 150, "spent": 200, "over_budget": 1},
    {"budget": 300, "spent": 250, "over_budget": 0},
    {"budget": 100, "spent": 100, "over_budget": 0},
    {"budget": 10000, "spent": 14000, "over_budget": 1},
    {"budget": 200000, "spent": 180000, "over_budget": 0},
    {"budget": 15000, "spent": 20000, "over_budget": 1},
    {"budget": 500000, "spent": 250000, "over_budget": 0},
    {"budget": 300000, "spent": 500000, "over_budget": 1},
    {"budget": 10000, "spent": 10000, "over_budget": 0},
])

# Features and target
X = data[["budget", "spent"]]
y = data["over_budget"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
joblib.dump(model, "model.joblib")
print("✅ Model trained and saved as model.joblib")
