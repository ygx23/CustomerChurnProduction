from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="Customer Churn Prediction API")
model_path= "model/churn_pipeline.pkl"

if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    raise FileNotFoundError(f'{model_path} not found')

class CustomerFeatures(BaseModel):
    tenure: float
    support_calls: int
    payment_delay: float
    total_spend: float
    usage_per_tenure: float

    is_unhappy: int
    is_high_value: int
    is_inactive: int

    gender: str
    contract_length: str
    age_bucket: str


@app.get("/")
def home():
    return {"message": "churn API is running. Go to /docs for the UI."}

@app.post("/predict")
def predict_churn(customer: CustomerFeatures):
    data_dict = customer.model_dump()
    input_df = pd.DataFrame([data_dict])
    churn_prob = model.predict_proba(input_df)[0][1]
    prediction = int(churn_prob >= 0.5)

    return {
        "churn_probability": round(float(churn_prob), 4),
        "churn_prediction": prediction,
        "decision": "retention alert" if prediction == 1 else "no action"}