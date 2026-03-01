import mlflow
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


# Get latest run ID programmatically
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name("Default")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["start_time DESC"],
    max_results=1,
)

latest_run_id = runs[0].info.run_id

# Load model from latest run
model_uri = f"runs:/{latest_run_id}/model"
model = mlflow.sklearn.load_model(model_uri)

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
