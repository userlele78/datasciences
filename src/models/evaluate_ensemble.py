import pandas as pd
import numpy as np
import joblib
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.utils.metrics import calculate_metrics
from src.models.train_ensemble import FuelPriceEnsemble

DATA_PATH = "f:/Project/DataSciences/data/processed/feature_dataset.csv"

def evaluate_ensemble():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=['date', 'target_delta'])
    y = df['target_delta']
    
    start_idx = 2000
    step_size = 30
    
    all_actuals = []
    all_preds = []
    
    print("Evaluating ENSEMBLE (RF + LGBM) via Walk-forward...")
    
    for i in range(start_idx, len(df), step_size):
        train_x, train_y = X.iloc[:i], y.iloc[:i]
        test_x, test_y = X.iloc[i:i+step_size], y.iloc[i:i+step_size]
        
        if len(test_x) == 0: break
        
        # We re-train in the walk-forward to be fair to other models
        model = FuelPriceEnsemble()
        model.fit(train_x, train_y)
        pred_delta = model.predict(test_x)
        
        current_prices = test_x['price_lag_0'].values
        all_preds.extend(current_prices + pred_delta)
        all_actuals.extend(current_prices + test_y.values)
        
    metrics = calculate_metrics(np.array(all_actuals), np.array(all_preds))
    
    print("\n" + "="*30)
    print("ENSEMBLE EVALUATION RESULT")
    print("="*30)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    print("\nSo sánh với mốc cũ (RandomForest):")
    print("MAE cũ: 65.0043")
    improvement = (65.0043 - metrics['MAE']) / 65.0043 * 100
    print(f"Mức độ cải thiện MAE: {improvement:.2f}%")

if __name__ == "__main__":
    evaluate_ensemble()
