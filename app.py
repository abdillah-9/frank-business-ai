from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn
import random
from faridaAI import router as farida_router

# Initialize FastAPI app
app = FastAPI()

# Include / register faridaAI.py routes
app.include_router(farida_router)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://frank-business-app.netlify.app",
        "https://woman-reproduction-health-tracker.netlify.app",
        "http://localhost:3000"
    ],
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

# GET endpoint to keep server alive
@app.get("/wakeUp")
def root():
    return {"status": "OK"}

@app.post("/predict")
def predict(data: List[BudgetItem]):
    response = []

# Tip templates
over_budget_titles = [
    "You've Gone Over Your Budget—Time to Reassess Your Spending Habits",
    "Budget Limit Exceeded: It's Crucial to Make Immediate Adjustments",
    "Your Spending Has Surpassed Safe Levels—Take Preventive Action Now",
    "Oops! You've Exceeded Your Budget—Consider Revisiting Your Financial Plan",
    "Budget Status: Over the Threshold—Urgent Attention Needed",
    "Warning: Your Financial Activity Has Pushed You Into the Red Zone",
    "Alert: Current Spending Patterns Have Breached Your Budget Cap",
    "Budget Violation Detected—A Strategic Review Is Strongly Advised",
    "Heads-Up! You're Spending More Than You Planned—Act Now to Regain Control",
    "High Spending Alert: It Looks Like You've Gone Over Your Intended Budget"
]

over_budget_descs = [
    "Your expenses have exceeded the limits you previously set. It might be a good time to review your discretionary spending and find areas where you can cut back.",
    "It appears that you're spending more than anticipated. Consider reevaluating your monthly expenses to identify what can be reduced or postponed.",
    "Your current spending habits are pushing your budget beyond its capacity. Trimming down non-essential purchases could help you regain financial control.",
    "Your budget has been exceeded, which may impact your financial goals. A quick review and adjustment of your spending plan is highly recommended.",
    "Spending is running higher than planned. This could affect savings or other financial goals, so look for areas where you can save.",
    "You've crossed your financial boundaries for the period. Tightening your spending on luxuries or optional services could help rebalance your budget.",
    "You've passed your set limit, suggesting that you might need to prioritize your essential expenses over discretionary ones for the rest of the period.",
    "This level of spending may lead to long-term budget imbalances. A deep dive into your expenditures might reveal ways to trim unnecessary costs.",
    "You've spent more than your planned budget allows. Reallocating funds and adjusting upcoming expenses could help steer things back on track.",
    "Your spending trends suggest a budget overrun. Consider using tools or alerts to stay more closely aligned with your financial goals moving forward."
]

warning_titles = [
    "Budget Caution: You're Halfway Through Your Limit—Time for Careful Monitoring",
    "Spending Alert: You've Reached the Midpoint of Your Budget—Stay Aware",
    "Approaching Your Spending Limit—It's Wise to Stay Vigilant",
    "Heads Up! Your Budget Usage Has Crossed the 50% Mark",
    "Caution: You're on a Steady Spending Trajectory—Monitor Closely",
    "Budget Check-In: You're at the Halfway Point—Consider Reevaluating",
    "You're on Track But Getting Close—Stay Alert to Avoid Overruns",
    "Budget Utilization Is Rising—Maintain Smart and Conscious Spending",
    "Halfway There Financially—It's a Good Time for a Midpoint Review",
    "Spending is Stable for Now, but Keep a Watchful Eye on What's Next"
]

warning_descs = [
    "You're at a critical point where you've used about half of your allocated budget. It's a good time to pause and assess how you're allocating your remaining funds.",
    "Your spending is currently moderate, but you're halfway to your limit. Consider reducing any planned non-essential purchases.",
    "You're doing okay, but it's important to think ahead. If spending continues at this rate, you might exceed your budget by the end of the cycle.",
    "At this halfway milestone, taking a moment to plan your remaining budget could help you stay within bounds.",
    "Your financial habits are steady so far, but it's wise to keep a watchful eye and avoid unnecessary splurges.",
    "You've spent about half of your budget already. Try to project what remaining expenses might look like and plan accordingly.",
    "Still on track, but now is the perfect time for a spending check-in. Make adjustments if necessary to avoid overspending.",
    "Your current pace is fine, but keep a close eye on recurring or unexpected expenses that might push you over the line.",
    "Spending is approaching a more sensitive range. Keeping a tighter grip now can help avoid surprises later.",
    "It's not urgent yet, but reviewing your financial priorities now can go a long way in ensuring a smooth rest of the month."
]

within_budget_titles = [
    "Congratulations! You're Currently Operating Well Within Your Budget",
    "All Systems Go: Your Spending Is Fully Under Control and On Track",
    "Excellent Work Staying Within Budget—You're Managing Finances Like a Pro",
    "Great News: Your Budget Is Healthy and Spending Is Well Balanced",
    "You're On Target Financially—Keep Up the Good Work!",
    "All Clear: Your Current Budget Is Balanced and Functioning Smoothly",
    "Spending Check Complete—Your Budget Is in Great Shape",
    "Smart Financial Habits Detected—You're Spending Responsibly",
    "Financial Status: Healthy and Sustainable Spending in Progress",
    "Nice Budgeting! Everything Is Aligned With Your Financial Goals"
]

within_budget_descs = [
    "You're doing a great job of keeping your expenses within your planned budget. Maintaining this pace will help you build long-term financial stability.",
    "Your current financial behavior shows excellent budgeting discipline. Keep it up, and you'll likely reach your savings goals with ease.",
    "You've demonstrated strong control over your spending. Staying consistent will help you avoid last-minute budget surprises.",
    "Everything looks great—your financial plan is working exactly as intended. Continue with your current habits for ongoing success.",
    "You're managing your finances well, which means you're in a good place to either save more or invest in something meaningful.",
    "Well done! You're not only staying within your limits but also showing that you're making smart, conscious financial decisions.",
    "Your spending aligns perfectly with your plan. This level of control will serve you well in both the short and long term.",
    "You've made sound financial decisions this period. Staying within your budget keeps things stress-free and predictable.",
    "This is a great example of healthy budgeting. If you continue this trend, you're setting yourself up for financial resilience.",
    "You're doing everything right financially—keep tracking your progress, and success will follow."
]

# Expense detail phrases
expense_phrases = [
    " Your largest recorded expense for this budget cycle was '{name}', which amounted to a significant total of {amount:,}Tsh. This single item had the biggest impact on your overall spending.",
    " The most substantial financial outlay this period came from '{name}', costing you {amount:,}Tsh. It's a good idea to review this item if you're looking to cut down expenses.",
    " Your top expenditure was the item '{name}', setting you back {amount:,}Tsh. As your biggest cost, it might be worth examining its necessity or frequency.",
    " You spent a notable {amount:,}Tsh on '{name}', making it your highest single expense during this budgeting period. Consider whether similar costs are essential in the future.",
    " The standout expense in this budgeting cycle was '{name}', which totaled {amount:,}Tsh. This item contributed the most to your overall spending this time around."
]


    for item in data:
        budget = item.budgetAmount
        spent = item.expenseTotal

        # Calculate percentage
        percentage_spent = round((spent / budget) * 100, 2) if budget else 0

        # Expense details
        if item.allExpenses:
            highest_expense = max(item.allExpenses, key=lambda x: x.amount)
            phrase = random.choice(expense_phrases)
            expense_desc = phrase.format(name=highest_expense.name, amount=highest_expense.amount)
        else:
            expense_desc = ""

        # Tips and flag logic
        if percentage_spent > 75:
            tip_title = random.choice(over_budget_titles)
            tip_desc = random.choice(over_budget_descs) + expense_desc
            over_budget_flag = True
        elif 50 < percentage_spent <= 75:
            tip_title = random.choice(warning_titles)
            tip_desc = random.choice(warning_descs) + expense_desc
            over_budget_flag = False
        else:
            tip_title = random.choice(within_budget_titles)
            tip_desc = random.choice(within_budget_descs) + expense_desc
            over_budget_flag = False

        # Build response item
        response.append({
            "budgetID": item.budgetID,
            "budgetName": item.budgetName,
            "tip_title": tip_title,
            "tip_desc": tip_desc,
            "over_budget": over_budget_flag,
            "percentage_spent": percentage_spent
        })

    return response

# Run the FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
