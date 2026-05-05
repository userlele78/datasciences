from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.models.train_ensemble import FuelPriceEnsemble

app = FastAPI(title="Fuel Price Forecasting API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load all models
MODELS_DIR = "f:/Project/DataSciences/models"
models = {}

model_files = {
    "random_forest": "final_rf_model.pkl",
    "xgboost": "xgboost_delta_v2.pkl",
    "ensemble": "ensemble_model.pkl"
}

for name, filename in model_files.items():
    path = os.path.join(MODELS_DIR, filename)
    if os.path.exists(path):
        models[name] = joblib.load(path)
        print(f"Loaded model: {name}")

class PriceFeatures(BaseModel):
    price_lag_0: float
    brent_lag_0: float
    brent_lag_7: float
    brent_mean_30: float
    regime: int
    brent_diff_1: float
    brent_diff_7: float
    days_since_last_change: int
    model_name: str = "random_forest"

@app.get("/")
def home():
    return {"available_models": list(models.keys()), "status": "Ready"}

@app.post("/predict")
def predict(features: PriceFeatures):
    if features.model_name not in models:
        raise HTTPException(status_code=400, detail=f"Model {features.model_name} not available")
    
    try:
        model = models[features.model_name]
        # Prepare data (exclude model_name)
        data_dict = features.model_dump()
        del data_dict['model_name']
        
        input_data = pd.DataFrame([data_dict])
        delta_pred = model.predict(input_data)[0]
        final_price = features.price_lag_0 + delta_pred
        
        return {
            "model_used": features.model_name,
            "predicted_delta": round(float(delta_pred), 2),
            "predicted_price_tomorrow": round(float(final_price), 2),
            "current_price": features.price_lag_0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
