# train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import random

# Seed for reproducibility
random.seed(42)

# Dynamically generate training data
training_data = []

for i in range(500):  # Generate 500 examples
    budget = random.randint(1_000, 500_000) # the _ is just ignored, its just for formating 
    
    # Randomly vary spent by ±50% of budget
    spent = budget + random.randint(-int(budget * 0.5), int(budget * 0.5))

    # New rule: over_budget if spent >= 75% of budget
    if spent >= 0.75 * budget:
        over_budget = 1
    else:
        over_budget = 0

    training_data.append({
        "budget": budget,
        "spent": spent,
        "over_budget": over_budget
    })

# Convert to DataFrame
df = pd.DataFrame(training_data)

# Split into features and target
X = df[["budget", "spent"]]
y = df["over_budget"]

# Use a pipeline with a scaler
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(random_state=42))
])

# Train the model
pipeline.fit(X, y)

# Save the model
joblib.dump(pipeline, "model.joblib")

print("✅ Realistic model trained and saved as model.joblib")
