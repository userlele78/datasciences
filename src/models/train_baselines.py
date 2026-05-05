import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.utils.validation import walk_forward_validation
from src.utils.metrics import calculate_metrics

def train_baselines(data_path):
    df = pd.read_csv(data_path)
    naive_preds = df['price_lag_0'].values[2000:] 
    actuals = df['target_next_day'].values[2000:]
    
    print("--- NAIVE BASELINE ---")
    print(calculate_metrics(actuals, naive_preds))

    def lr_model(train_x, train_y, test_x):
        model = LinearRegression()
        model.fit(train_x, train_y)
        return model.predict(test_x)

    print("\nRunning Walk-Forward for Linear Regression...")
    act_lr, pre_lr = walk_forward_validation(df, lr_model, start_idx=2000, step_size=30)
    print("--- LINEAR REGRESSION ---")
    print(calculate_metrics(act_lr, pre_lr))

if __name__ == "__main__":
    DATA = "f:/Project/DataSciences/data/processed/feature_dataset.csv"
    train_baselines(DATA)
