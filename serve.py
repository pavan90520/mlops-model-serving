import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

# Load model from project root
model = joblib.load("model.pkl")

app = FastAPI(title="ML Model Serving API")


class PredictionRequest(BaseModel):
    features: list


@app.post("/predict")
def predict(data: PredictionRequest):
    df = pd.DataFrame([data.features])
    proba = model.predict_proba(df)[0]
    prediction = model.predict(df)[0]

    return {
        "prediction": int(prediction),
        "probability_benign": float(proba[1]),
        "probability_malignant": float(proba[0])
    }
