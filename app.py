from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import json
import numpy as np

app = FastAPI()

# Load model and metrics
model = joblib.load('diabetes_model.pkl')
with open('metrics.json', 'r') as f:
    metrics = json.load(f)

class PatientData(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(data: PatientData):
    input_data = np.array([[data.Pregnancies, data.Glucose, data.BloodPressure, data.SkinThickness,
                            data.Insulin, data.BMI, data.DiabetesPedigreeFunction, data.Age]])
    prediction = model.predict(input_data)[0]
    confidence = model.predict_proba(input_data)[0][prediction]
    result = "Diabetic" if prediction == 1 else "Not Diabetic"
    return {"prediction": int(prediction), "result": result, "confidence": float(confidence)}

@app.get("/metrics")
async def get_metrics():
    return metrics