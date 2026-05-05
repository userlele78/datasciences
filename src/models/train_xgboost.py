import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.utils.metrics import calculate_metrics

DATA_PATH = "f:/Project/DataSciences/data/processed/feature_dataset.csv"
MODEL_DIR = "f:/Project/DataSciences/models"

def train_delta_model():
    df = pd.read_csv(DATA_PATH)
    
    # Features (excluding date and delta)
    X = df.drop(columns=['date', 'target_delta'])
    y = df['target_delta']
    
    # Custom Walk-forward for Delta
    start_idx = 2000
    step_size = 30
    
    all_actual_levels = []
    all_pred_levels = []
    
    for i in range(start_idx, len(df), step_size):
        train_x = X.iloc[:i]
        train_y = y.iloc[:i]
        
        test_x = X.iloc[i:i+step_size]
        test_y = y.iloc[i:i+step_size]
        
        if len(test_x) == 0: break
            
        model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42)
        model.fit(train_x, train_y)
        
        # Predict Delta
        pred_delta = model.predict(test_x)
        
        # Convert back to levels: Price(t) + predicted_delta
        # Price(t) is stored in 'price_lag_0'
        current_prices = test_x['price_lag_0'].values
        pred_levels = current_prices + pred_delta
        actual_levels = current_prices + test_y.values
        
        all_pred_levels.extend(pred_levels)
        all_actual_levels.extend(actual_levels)
        
    metrics = calculate_metrics(np.array(all_actual_levels), np.array(all_pred_levels))
    print("\n--- OPTIMIZED XGBOOST RESULTS (DELTA PREDICTION) ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # Save final model
    final_model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, random_state=42)
    final_model.fit(X, y)
    joblib.dump(final_model, os.path.join(MODEL_DIR, "xgboost_delta_v2.pkl"))
    print("\nOptimized Delta model saved.")

if __name__ == "__main__":
    train_delta_model()
