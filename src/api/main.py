from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import os
import sys

# Sử dụng đường dẫn tương đối dựa trên vị trí file main.py
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, "models")
sys.path.append(BASE_DIR)

from src.models.train_ensemble import FuelPriceEnsemble

app = FastAPI(title="Fuel Price Forecasting API", version="1.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load all models dynamically
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
    horizon: int = 1  # Dự báo bao nhiêu ngày tiếp theo

@app.get("/")
def home():
    return {"available_models": list(models.keys()), "status": "Ready", "base_dir": BASE_DIR}

@app.get("/samples")
def get_samples():
    path = os.path.join(BASE_DIR, "data/processed/feature_dataset.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    df = pd.read_csv(path)
    # Lấy 100 dòng cuối cùng
    samples = df.tail(100).to_dict(orient="records")
    return samples

@app.post("/predict")
def predict(features: PriceFeatures):
    if features.model_name not in models:
        raise HTTPException(status_code=400, detail=f"Model {features.model_name} not available")
    
    try:
        model = models[features.model_name]
        results = []
        current_price = features.price_lag_0
        
        # Tạo bản sao dữ liệu để dự báo đệ quy (Recursive)
        temp_features = features.model_dump()
        del temp_features['model_name']
        del temp_features['horizon']
        
        for i in range(features.horizon):
            input_df = pd.DataFrame([temp_features])
            delta_pred = model.predict(input_df)[0]
            next_price = current_price + delta_pred
            
            results.append({
                "day": i + 1,
                "predicted_delta": round(float(delta_pred), 2),
                "predicted_price": round(float(next_price), 2)
            })
            
            # Cập nhật dữ liệu cho ngày tiếp theo (Giả định brent oil không đổi hoặc giảm nhẹ)
            current_price = next_price
            temp_features['price_lag_0'] = next_price
            temp_features['days_since_last_change'] += 1
            
        return {
            "model_used": features.model_name,
            "horizon": features.horizon,
            "forecast": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
