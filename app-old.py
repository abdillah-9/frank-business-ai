# app.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import uvicorn

# Load the model
model = joblib.load("model.joblib")

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input schema
class BudgetInput(BaseModel):
    budget: float
    spent: float

# Output route
@app.post("/predict")
def predict(data: BudgetInput):
    X = [[data.budget, data.spent]]
    prediction = model.predict(X)[0]
    if prediction == 1:
        tip = "You're over budget. Consider reducing spending in this category."
    else:
        tip = "Your spending is within the budget. Keep it up!"
    return {"tip": tip, "over_budget": bool(prediction)}

# Run app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
