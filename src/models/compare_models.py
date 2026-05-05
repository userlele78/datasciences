import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.utils.metrics import calculate_metrics
from src.models.train_ensemble import FuelPriceEnsemble

DATA_PATH = "f:/Project/DataSciences/data/processed/feature_dataset.csv"

def run_comparison():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=['date', 'target_delta'])
    y = df['target_delta']
    
    start_idx = 2000
    step_size = 30
    
    # Danh sách các mô hình bao gồm cả Ensemble
    models = {
        "XGBoost": xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=42),
        "LightGBM": lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42, verbose=-1),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Ensemble (RF+LGBM)": FuelPriceEnsemble()
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"Evaluating {name}...")
        all_actuals = []
        all_preds = []
        
        for i in range(start_idx, len(df), step_size):
            train_x, train_y = X.iloc[:i], y.iloc[:i]
            test_x, test_y = X.iloc[i:i+step_size], y.iloc[i:i+step_size]
            
            if len(test_x) == 0: break
            
            model.fit(train_x, train_y)
            pred_delta = model.predict(test_x)
            
            # Convert back to levels
            current_prices = test_x['price_lag_0'].values
            all_preds.extend(current_prices + pred_delta)
            all_actuals.extend(current_prices + test_y.values)
            
        results[name] = calculate_metrics(np.array(all_actuals), np.array(all_preds))

    # SARIMAX (giữ nguyên để tham khảo)
    print("Evaluating SARIMAX...")
    ts_data = df['price_lag_0']
    exo_data = df[['brent_lag_0']]
    sarimax_actuals = []
    sarimax_preds = []
    
    for i in range(start_idx, len(df), step_size):
        train_y = ts_data.iloc[:i]
        train_exo = exo_data.iloc[:i]
        test_y = ts_data.iloc[i:i+step_size]
        test_exo = exo_data.iloc[i:i+step_size]
        if len(test_y) == 0: break
        try:
            model_sm = SARIMAX(train_y, exog=train_exo, order=(1,1,1))
            res = model_sm.fit(disp=False)
            preds = res.forecast(steps=len(test_y), exog=test_exo)
            sarimax_preds.extend(preds)
            sarimax_actuals.extend(test_y.values)
        except: continue
    results["SARIMAX"] = calculate_metrics(np.array(sarimax_actuals), np.array(sarimax_preds))

    # Summary
    print("\n" + "="*30)
    print("FINAL COMPARISON REPORT (WITH ENSEMBLE)")
    print("="*30)
    report_df = pd.DataFrame(results).T
    print(report_df)
    
    report_df.to_csv("f:/Project/DataSciences/logs/model_comparison_final.csv")

if __name__ == "__main__":
    run_comparison()
