# train_model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import random

# For reproducibility
random.seed(42)

# Generate synthetic training data
training_data = []

def generate_example(budget_range):
    budget = random.randint(*budget_range)
    variation = random.randint(-int(budget * 0.5), int(budget * 0.5))
    spent = max(0, budget + variation)  # Ensure spent is not negative
    percentage_spent = spent / budget if budget > 0 else 0
    over_budget = 1 if percentage_spent >= 0.75 else 0

    return {
        "budget": budget,
        "spent": spent,
        "percentage_spent": percentage_spent,
        "over_budget": over_budget
    }

# Generate 500 examples: include small, medium, and large budgets
for _ in range(200):  # Small budgets (1k–5k)
    training_data.append(generate_example((1000, 5000)))

for _ in range(150):  # Medium budgets (5k–50k)
    training_data.append(generate_example((5001, 50000)))

for _ in range(150):  # Large budgets (50k–500k)
    training_data.append(generate_example((50001, 500000)))

# Create DataFrame
df = pd.DataFrame(training_data)

# Shuffle to mix small/large examples
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Features and target
X = df[["budget", "spent", "percentage_spent"]]
y = df["over_budget"]

# Define pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(random_state=42))
])

# Train model
pipeline.fit(X, y)

# Save model
joblib.dump(pipeline, "model.joblib")

print("✅ Model trained and saved as model.joblib (with small budgets and spent)")
