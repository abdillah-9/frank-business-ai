from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import joblib
import uvicorn
import random
from faridaAI import router as farida_router

# Initialize FastAPI app
app = FastAPI()

#include /register faridaAI.py routes
app.include_router(farida_router)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://frank-business-app.netlify.app",
        "https://woman-reproduction-health-tracker.netlify.app"
    ]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the schema for each expense
class Expense(BaseModel):
    name: str
    amount: int

# Define the schema for each budget item, including allExpenses
class BudgetItem(BaseModel):
    budgetID: int
    budgetName: str
    budgetAmount: float
    expenseTotal: float
    allExpenses: List[Expense]

# Load your ML model
model = joblib.load("model.joblib")  # Update with your actual model path

# POST endpoint to predict if a budget is over or under
@app.get("/wakeUp")
def root():
    return {"status": "OK"}

@app.post("/predict")
def predict(data: List[BudgetItem]):
    response = []

    # Tip templates
    over_budget_titles = [
        "Budget exceeded", "Over the limit", "Spending too high", "Exceeded your budget",
        "Above budget", "You're in the red", "Overspending alert", "Budget breach",
        "Watch your spending", "Overage detected"
    ]

    over_budget_descs = [
        "Consider reducing spending.", "Reevaluate your expenses.",
        "Try cutting down on non-essentials.", "You may want to adjust your budget.",
        "Spending is high—look for savings.", "Time to tighten the belt a bit.",
        "You've passed your limit—try reviewing priorities.",
        "Your budget needs attention. A review might help.",
        "Spending more than planned—consider rebalancing.",
        "Costs are creeping up—time for a course correction."
    ]

    within_budget_titles = [
        "Within budget", "On track", "Budget is healthy", "Under control",
        "Spending in check", "All good here", "Budget balanced", "Smart spending",
        "Financially fit", "Nice budgeting!"
    ]

    within_budget_descs = [
        "Your spending is within the budget. Keep it up!", "Great job staying on track!",
        "You're managing your finances well.", "Keep up the responsible spending.",
        "Well done—you're staying within limits!", "Nice work—your budget is under control.",
        "You're right on the money—literally!", "Everything looks balanced. Smart spending!",
        "You're budgeting like a pro.", "Finances are looking healthy. Keep going!"
    ]

    # Randomized expense explanation templates
    expense_phrases = [
        " Your highest expense is '{name}' which cost {amount:,}Tsh",
        " The most significant cost was '{name}' at {amount:,}Tsh",
        " You spent {amount:,}Tsh on '{name}', which was your biggest item.",
        " '{name}' cost you {amount:,}Tsh, the highest in this budget.",
        " The top expense in this budget was '{name}' totaling {amount:,}Tsh"
    ]

    # Process each budget item
    for item in data:
        budget = item.budgetAmount
        spent = item.expenseTotal

        # Predict using the model
        prediction = model.predict([[budget, spent]])[0]

        # Determine highest expense
        if item.allExpenses:
            highest_expense = max(item.allExpenses, key=lambda x: x.amount)
            phrase = random.choice(expense_phrases)
            expense_desc = phrase.format(name=highest_expense.name, amount=highest_expense.amount)
        else:
            expense_desc = ""

        # Generate final tip
        if prediction == 1:
            tip_title = random.choice(over_budget_titles)
            tip_desc = random.choice(over_budget_descs) + expense_desc
        else:
            tip_title = random.choice(within_budget_titles)
            tip_desc = random.choice(within_budget_descs) + expense_desc

        # Add to response
        response.append({
            "budgetID": item.budgetID,
            "budgetName": item.budgetName,
            "tip_title": tip_title,
            "tip_desc": tip_desc,
            "over_budget": bool(prediction)
        })

    return response

# Run the FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
