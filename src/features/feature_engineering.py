import pandas as pd
import numpy as np
import os
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def generate_features_optimized(input_path, output_path):
    df = pd.read_csv(input_path)
    df['date'] = pd.to_datetime(df['date'])
    target = 'gas_ron95'
    
    # 1. Regime Detection
    df['vol_30'] = df[target].rolling(window=30).std()
    regime_data = df[[target, 'vol_30']].bfill()
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['regime'] = kmeans.fit_predict(StandardScaler().fit_transform(regime_data))
    
    features = pd.DataFrame(index=df.index)
    features['date'] = df['date']
    
    # Standard Features
    features['price_lag_0'] = df[target]
    features['brent_lag_0'] = df['brent_oil']
    features['brent_lag_7'] = df['brent_oil'].shift(7)
    features['brent_mean_30'] = df['brent_oil'].rolling(window=30).mean()
    features['regime'] = df['regime']
    
    # MOMENTUM FEATURES (Crucial for Delta prediction)
    features['brent_diff_1'] = df['brent_oil'].diff(1)
    features['brent_diff_7'] = df['brent_oil'].diff(7)
    
    # NEW FEATURE: Days since last price change
    # We detect when price changed and count days
    price_changed = (df[target].diff() != 0).astype(int)
    days_since_change = []
    count = 0
    for changed in price_changed:
        if changed == 1: count = 0
        else: count += 1
        days_since_change.append(count)
    features['days_since_last_change'] = days_since_change
    
    # TARGET: Delta y = y(t+1) - y(t)
    features['target_delta'] = df[target].shift(-1) - df[target]
    
    final_df = features.dropna()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False)
    print(f"Optimized features saved to {output_path}")

if __name__ == "__main__":
    CLEAN_DATA = "f:/Project/DataSciences/data/processed/clean_data.csv"
    FEATURE_DATA = "f:/Project/DataSciences/data/processed/feature_dataset.csv"
    generate_features_optimized(CLEAN_DATA, FEATURE_DATA)
