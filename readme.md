# MLOps Project – Model Training & API Serving

This project demonstrates an end-to-end MLOps workflow:
- Model training
- Experiment tracking with MLflow
- Model serving via FastAPI

## Tech Stack
- Python
- scikit-learn
- MLflow
- FastAPI
- Uvicorn

## How to run
1. Install dependencies: `pip install -r requirements.txt`
2. Train model: `python train.py`
3. Start API: `uvicorn serve:app --reload`
4. Test at: http://127.0.0.1:8000/docs
