import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# ✅ Use the correct run ID
RUN_ID = "ab600f979a9d4624bf6f1d6bdce02af9"

# Load model directly from mlruns folder
model = mlflow.sklearn.load_model(
    f"mlruns/0/{RUN_ID}/artifacts/model"
)

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