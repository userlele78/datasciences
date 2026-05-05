import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import joblib
import os
import sys

# Paths
MODEL_PATH = "f:/Project/DataSciences/models/xgboost_v1.pkl"
DATA_PATH = "f:/Project/DataSciences/data/processed/feature_dataset.csv"
OUTPUT_DIR = "f:/Project/DataSciences/logs"

def analyze_importance():
    if not os.path.exists(MODEL_PATH):
        print("Model not found.")
        return
        
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    features = df.drop(columns=['date', 'target_next_day']).columns
    
    # Get importance
    importance = model.feature_importances_
    feat_imp = pd.Series(importance, index=features).sort_values(ascending=False)
    
    print("\n--- FEATURE IMPORTANCE ---")
    print(feat_imp)
    
    plt.figure(figsize=(10, 6))
    feat_imp.plot(kind='bar')
    plt.title("XGBoost Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"))
    plt.close()
    
    with open(os.path.join(OUTPUT_DIR, "interpretability_log.txt"), "w", encoding='utf-8') as f:
        f.write("FEATURE IMPORTANCE LOG\n")
        f.write(feat_imp.to_string())
    
    print(f"\nAnalysis saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    analyze_importance()
