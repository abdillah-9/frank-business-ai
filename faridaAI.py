from fastapi import FastAPI
from fastapi import APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import joblib
import uvicorn
import numpy as np
import random
from datetime import datetime, timedelta

# Load models
ovulationModel = joblib.load("ovulationFaridaModel.joblib")
pregnancyModel = joblib.load("pregnancyFaridaModel.joblib")
childcareModel = joblib.load("childcareModel.joblib")

STRESS_MAP = {"low": 0, "medium": 1, "high": 2}

router = APIRouter()

# ----------- Data Models -----------

class OvulationRecord(BaseModel):
    name: str
    age: int
    last_period_date: str
    cycle_length_days: int
    period_duration_days: int
    is_cycle_regular: bool
    stress_level: str
    sleep_hours: int
    day_week_exercise: int
    diagnosed_conditions: str
    userID: str
    status: str

class OvulationRequest(BaseModel):
    filteredOvulation: List[OvulationRecord]

class PregnanceRecord(BaseModel):
    id: int
    created_at: str
    pregnance_week: int
    featus_number: int
    is_smoking: bool
    is_drinking: bool
    mental_health_problem: bool
    symptoms: str
    prev_pregnancy_issues: str
    fetal_HR: int
    mother_HR: int
    status: str
    userID: str

class PregnanceRequest(BaseModel):
    filteredPregnance: List[PregnanceRecord]

class ChildcareRecord(BaseModel):
    id: int
    created_at: str
    userID: str
    baby_age_month: int
    gender: str
    birth_weight: float
    current_weight: float
    feeding_type: str
    feeding_frequency: int
    sleep_hours: int
    status: str

class ChildcareRequest(BaseModel):
    filteredChildcare: List[ChildcareRecord]

# ----------- Utility Functions -----------

def calculate_ovulation_date(last_period_date: str, cycle_length: int = 28):
    try:
        start_date = datetime.strptime(last_period_date, "%Y-%m-%d")
        ovulation_date = start_date + timedelta(days=(cycle_length - 14))
        return ovulation_date
    except Exception:
        return None

def format_date(dt: datetime):
    if dt:
        return dt.strftime("%Y-%m-%d")
    return None

# ----------- Routes -----------

@router.get("/wakeUp")
def wake_up():
    return {"status": "OK"}

# ----------- Ovulation Logic -----------

def probabilistic_score(record):
    score = 0
    score += 2 if record["is_cycle_regular"] else 0
    score += 1 if record["sleep_hours"] >= 6 else 0
    score += {"low": 2, "medium": 1, "high": 0}[record["stress_level"]]
    score += 1 if record["day_week_exercise"] >= 1 else 0
    return 1 if score >= 4 else 0

def predict_user_condition(user_data: List[dict]):
    if len(user_data) < 3:
        return probabilistic_score(user_data[-1])

    avg_sleep = np.mean([d['sleep_hours'] for d in user_data])
    avg_exercise = np.mean([d['day_week_exercise'] for d in user_data])
    cycle_std = np.std([d['cycle_length_days'] for d in user_data])
    regular_ratio = sum(d['is_cycle_regular'] for d in user_data) / len(user_data)
    avg_stress = np.mean([STRESS_MAP[d['stress_level']] for d in user_data])

    features = [[
        user_data[-1]["age"],
        regular_ratio,
        avg_stress,
        avg_sleep,
        avg_exercise,
        cycle_std,
    ]]

    return int(ovulationModel.predict(features)[0])

@router.post("/farida-ovulation-api")
def predict_ovulation(request: OvulationRequest):
    user_data = [record.dict() for record in request.filteredOvulation]
    if not user_data:
        return {"error": "No ovulation data provided."}

    condition_ok = predict_user_condition(user_data)
    latest = user_data[-1]
    ovulation_dt = calculate_ovulation_date(
        latest["last_period_date"], latest["cycle_length_days"]
    )

    if ovulation_dt:
        fertile_window_start = ovulation_dt - timedelta(days=5)
        fertile_window_end = ovulation_dt
        next_period_dt = ovulation_dt + timedelta(days=14)
    else:
        fertile_window_start = fertile_window_end = next_period_dt = None

    # Pregnancy chance prediction
    pregnancy_features = [[
        latest["age"],
        1 if latest["is_cycle_regular"] else 0,
        STRESS_MAP.get(latest["stress_level"], 1),
        latest["sleep_hours"],
        latest["day_week_exercise"],
        latest["cycle_length_days"],
        latest["period_duration_days"],
    ]]
    pregnancy_chance = pregnancyModel.predict_proba(pregnancy_features)[0][1]

    hint_list = []

    if not latest["is_cycle_regular"]:
        hint_list.append(random.choice([
            "Cycle is irregular. Consider tracking it or consulting a doctor.",
            "Irregular cycles may signal hormonal changes — consult a specialist.",
            "Tracking irregular cycles helps predict ovulation better."
        ]))

    if latest["sleep_hours"] < 6:
        hint_list.append(random.choice([
            "Try to get at least 6-8 hours of sleep.",
            "Your sleep seems low — aim for restful nights 💤",
            "Sleep is vital for hormonal balance. Improve it! 😴"
        ]))

    if latest["stress_level"] == "high":
        hint_list.append(random.choice([
            "Your stress level is high. Consider stress-relief activities.",
            "Try yoga, meditation, or light walks to reduce stress.",
            "Stress affects ovulation. Try to unwind and relax 💆‍♀️"
        ]))

    if latest["day_week_exercise"] < 2:
        hint_list.append(random.choice([
            "Increase exercise to at least 2-3 times a week.",
            "Staying active supports reproductive health!",
            "Light exercise boosts circulation and hormones 🏃‍♀️"
        ]))

    final_hint = random.choice(hint_list) if hint_list else "Great job! You're maintaining healthy habits. ✅"

    return {
        "userID": latest["userID"],
        "name": latest["name"],
        "condition_ok": condition_ok,
        "message": random.choice([
            "All looks good ✅",
            "Healthy signs! Keep it up 🌸",
            "Ovulation health is in a good place 👍"
        ]) if condition_ok else random.choice([
            "Something may need attention ⚠️",
            "Ovulation pattern off. Please track or consult 🩺",
            "Some concerns detected. Monitor carefully 👩‍⚕️"
        ]),
        "hint": [final_hint],
        "predicted_ovulation_date": format_date(ovulation_dt),
        "fertile_window_start": format_date(fertile_window_start),
        "fertile_window_end": format_date(fertile_window_end),
        "predicted_next_period_date": format_date(next_period_dt),
        "predicted_pregnancy_chance": round(pregnancy_chance * 100, 2),  # % value
    }

# ----------- Pregnancy Logic -----------

def predict_pregnancy_condition(user_data: List[dict]):
    latest = user_data[-1]

    features = [[
        latest["pregnance_week"],
        latest["featus_number"],
        int(latest["is_smoking"]),
        int(latest["is_drinking"]),
        int(latest["mental_health_problem"]),
        latest["fetal_HR"],
        latest["mother_HR"]
    ]]

    return int(pregnancyModel.predict(features)[0])

@router.post("/farida-pregnancy-api")
def predict_pregnancy(request: PregnanceRequest):
    user_data = [record.dict() for record in request.filteredPregnance]
    if not user_data:
        return {"error": "No pregnancy data provided."}

    condition_ok = predict_pregnancy_condition(user_data)
    latest = user_data[-1]
    hint_list = []

    if latest["is_smoking"]:
        smoking_hints = [
            "Smoking during pregnancy can harm the baby.",
            "Avoid smoking — it's linked to preterm birth risks.",
            "Quit smoking to protect your baby’s lungs."
        ]
        hint_list.append(random.choice(smoking_hints))

    if latest["is_drinking"]:
        drinking_hints = [
            "Avoid alcohol to prevent fetal health risks.",
            "Even small amounts of alcohol may impact development.",
            "Pregnancy and alcohol don’t mix — stay safe."
        ]
        hint_list.append(random.choice(drinking_hints))

    if latest["mental_health_problem"]:
        mental_hints = [
            "Mental health support is important during pregnancy.",
            "Talk to someone — mental wellness helps both mom and baby.",
            "Seek mental health guidance if you feel overwhelmed."
        ]
        hint_list.append(random.choice(mental_hints))

    if latest["fetal_HR"] <= 0:
        hr_hints = [
            "No fetal heart rate recorded. Please monitor closely.",
            "Missing fetal HR — get it checked ASAP.",
            "Check fetal heartbeat to ensure baby’s well-being."
        ]
        hint_list.append(random.choice(hr_hints))

    if latest["mother_HR"] <= 0:
        mother_hr_hints = [
            "Mother's heart rate not recorded. Check with a professional.",
            "Record mother's HR regularly for safety.",
            "Missing maternal HR data — please follow up."
        ]
        hint_list.append(random.choice(mother_hr_hints))

    final_hint = random.choice(hint_list) if hint_list else "You're doing well! Keep following health guidelines. ✅"

    return {
        "userID": latest["userID"],
        "condition_ok": condition_ok,
        "message": random.choice([
            "Pregnancy health looks great ✅",
            "Stable signs detected — good job mama! 👶",
            "No serious issues seen — keep going strong!"
        ]) if condition_ok else random.choice([
            "Possible pregnancy concerns. Please consult ⚠️",
            "Watch out — irregular indicators detected 👩‍⚕️",
            "Some warning signs found — monitor closely 🩺"
        ]),
        "hint": [final_hint]
    }

# ----------- Childcare Logic -----------

def predict_childcare_condition(user_data: List[dict]):
    latest = user_data[-1]

    feeding_type_map = {"breastfeeding": 0, "formula": 1, "mixed": 2}
    gender_map = {"male": 0, "female": 1}

    features = [[
        latest["baby_age_month"],
        gender_map.get(latest["gender"].lower(), 0),
        latest["birth_weight"],
        latest["current_weight"],
        feeding_type_map.get(latest["feeding_type"].lower(), 0),
        latest["feeding_frequency"],
        latest["sleep_hours"],
    ]]

    return int(childcareModel.predict(features)[0])

@router.post("/farida-childcare-api")
def predict_childcare(request: ChildcareRequest):
    user_data = [record.dict() for record in request.filteredChildcare]
    if not user_data:
        return {"error": "No childcare data provided."}

    condition_ok = predict_childcare_condition(user_data)
    latest = user_data[-1]
    prevWeight = user_data[-2]["current_weight"] if len(user_data) > 1 else latest["current_weight"]
    hint_list = []

    if latest["current_weight"] < prevWeight:
        weight_hints = [
            "Baby's weight is dropping. Please consult a pediatrician.",
            "Growth check: baby's weight is lower than before.",
            "Unexpected weight loss. Visit your pediatric specialist."
        ]
        hint_list.append(random.choice(weight_hints))

    if latest["feeding_frequency"] < 8:
        feeding_hints = [
            "Feeding frequency is low. Ensure baby is fed 8-12 times a day.",
            "Newborns need frequent feeds — increase feeding sessions.",
            "Consider shorter intervals between feedings."
        ]
        hint_list.append(random.choice(feeding_hints))

    if latest["sleep_hours"] < 14:
        sleep_hints = [
            "Babies need at least 14-17 hours of sleep daily.",
            "Sleep is crucial for baby's brain development.",
            "Try to create a quiet sleep routine for the baby."
        ]
        hint_list.append(random.choice(sleep_hints))

    final_hint = random.choice(hint_list) if hint_list else "Baby’s growth and care patterns seem healthy ✅"

    return {
        "userID": latest["userID"],
        "condition_ok": condition_ok,
        "message": random.choice([
            "All childcare signs are healthy ✅",
            "Great progress — baby is on the right track 👶",
            "Health metrics look good — keep caring lovingly 💖"
        ]) if condition_ok else random.choice([
            "Care indicators show some concerns ⚠️",
            "Monitor baby’s growth more closely.",
            "Pediatrician check may be helpful 🩺"
        ]),
        "hint": [final_hint]
    }
