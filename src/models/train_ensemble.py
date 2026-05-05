import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Paths
from pathlib import Path
import os

# Lấy thư mục gốc project (datasciences/)
BASE_DIR = Path(__file__).resolve().parents[2]

# Đường dẫn data & model
DATA_PATH = BASE_DIR / "data" / "processed" / "feature_dataset.csv"
MODEL_DIR = BASE_DIR / "models"

# Tạo thư mục nếu chưa có
os.makedirs(MODEL_DIR, exist_ok=True)
class FuelPriceEnsemble:
    def __init__(self):
        self.rf = RandomForestRegressor(n_estimators=200, random_state=42)
        self.lgb = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42, verbose=-1)
        
    def fit(self, X, y):
        self.rf.fit(X, y)
        self.lgb.fit(X, y)
        
    def predict(self, X):
        # Weighted average: RF (60%) + LGBM (40%) based on MAE/RMSE results
        p_rf = self.rf.predict(X)
        p_lgb = self.lgb.predict(X)
        return 0.6 * p_rf + 0.4 * p_lgb

def train_and_save_ensemble():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=['date', 'target_delta'])
    y = df['target_delta']
    
    print("Training Final Ensemble Model (RF + LGBM)...")
    ensemble = FuelPriceEnsemble()
    ensemble.fit(X, y)
    
    # Save the custom object
    save_path = os.path.join(MODEL_DIR, "ensemble_model.pkl")
    joblib.dump(ensemble, save_path)
    print(f"Ensemble model saved to {save_path}")

if __name__ == "__main__":
    train_and_save_ensemble()
