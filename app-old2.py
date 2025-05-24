from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import joblib  # or the specific library you use for your model
import uvicorn
import random

# Initialize FastAPI app
app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the schema for each budget item
class BudgetItem(BaseModel):
    budgetID: int
    budgetName: str
    budgetAmount: float
    expenseTotal: float

# Load your ML model (assuming you have a model file)
model = joblib.load("model.joblib")  # Update this with the actual path to your model file

# POST endpoint to predict if a budget is over or under
@app.post("/predict")
def predict(data: List[BudgetItem]):
    # Create a response list to store results for each budget item
    response = []
    
    for item in data:
        budget = item.budgetAmount
        spent = item.expenseTotal
        
        # Predict based on the model
        prediction = model.predict([[budget, spent]])[0]  # Assuming the model expects a 2D array
        
        # Tip variations
        over_budget_titles = [
        "Budget exceeded",
        "Over the limit",
        "Spending too high",
        "Exceeded your budget",
        "Above budget",
        "You're in the red",
        "Overspending alert",
        "Budget breach",
        "Watch your spending",
        "Overage detected"
        ]


        over_budget_descs = [
        "Consider reducing spending.",
        "Reevaluate your expenses.",
        "Try cutting down on non-essentials.",
        "You may want to adjust your budget.",
        "Spending is high—look for savings.",
        "Time to tighten the belt a bit.",
        "You’ve passed your limit—try reviewing priorities.",
        "Your budget needs attention. A review might help.",
        "Spending more than planned—consider rebalancing.",
        "Costs are creeping up—time for a course correction."
        ]

        within_budget_titles = [
        "Within budget",
        "On track",
        "Budget is healthy",
        "Under control",
        "Spending in check",
        "All good here",
        "Budget balanced",
        "Smart spending",
        "Financially fit",
        "Nice budgeting!"
        ]

        within_budget_descs = [   
            "Your spending is within the budget. Keep it up!",
            "Great job staying on track!",
            "You're managing your finances well.",
            "Keep up the responsible spending.",
            "Well done—you're staying within limits!",
            "Nice work—your budget is under control.",
            "You're right on the money—literally!",
            "Everything looks balanced. Smart spending!",
            "You're budgeting like a pro.",
            "Finances are looking healthy. Keep going!"
        ]

        # Determine the tip based on prediction
        if prediction == 1:
            tip_title = random.choice(over_budget_titles)
            tip_desc = random.choice(over_budget_descs)
        else:
            tip_title = random.choice(within_budget_titles)
            tip_desc = random.choice(within_budget_descs)
        
        # Append the result to the response list
        response.append({
            "budgetID": item.budgetID,
            "budgetName": item.budgetName,
            "tip_title": tip_title,
            "tip_desc": tip_desc,
            "over_budget": bool(prediction)
        })
    
    # Return the response with tips for each budget
    return response

# Run the FastAPI app with Uvicorn (if this script is run directly)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
