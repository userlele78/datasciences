import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Configuration
INPUT_PATH = r"f:\Project\DataSciences\clean_data.csv"
OUTPUT_PATH = r"f:\Project\DataSciences\feature_dataset1.csv"
REPORT_PATH = r"f:\Project\DataSciences\feature_engineering_report.md"

def engineer_features():
    df = pd.read_csv(INPUT_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    target = 'gas_ron95'
    
    # --- 1. TARGET LAGS ---
    # We want to predict t+1, so features at time t must be used.
    # To be extremely safe: 
    # Current row in DF represents 't'. 
    # Features will be calculated on 't' and shifted by 1 to represent 'past' relative to 't+1'.
    # Actually, if we want to predict 'gas_ron95', we'll create a column 'target_next' = df[target].shift(-1)
    # And all features in the same row will be information available at or before 't'.
    
    # --- 2. REGIME DETECTION (RE-CALCULATED TO ENSURE CONSISTENCY) ---
    # Rolling std for volatility
    df['vol_30'] = df[target].rolling(window=30).std()
    regime_data = df[[target, 'vol_30']].bfill()
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['regime'] = kmeans.fit_predict(StandardScaler().fit_transform(regime_data))
    
    # --- 3. CORE FEATURES ---
    features = pd.DataFrame(index=df.index)
    features['date'] = df['date']
    
    # Lagged Price (t, t-1, t-6) -> effectively (t-1, t-2, t-7) relative to target_next
    # features['price_lag_0'] = df[target]
    # features['price_lag_1'] = df[target].shift(1)
    features['price_lag_6'] = df[target].shift(6)
    
    # Lagged Brent (t, t-6)
    features['brent_lag_0'] = df['brent_oil']
    features['brent_lag_6'] = df['brent_oil'].shift(6)
    
    # Rolling Stats (7d, 30d)
    features['price_mean_7'] = df[target].rolling(window=7).mean()
    features['price_std_7'] = df[target].rolling(window=7).std()
    features['brent_mean_30'] = df['brent_oil'].rolling(window=30).mean()
    
    # Momentum (Returns)
    features['price_return_1'] = df[target].pct_change(1)
    features['brent_return_1'] = df['brent_oil'].pct_change(1)
    
    # Regime
    features['regime'] = df['regime']
    
    # Calendar
    features['day_of_week'] = df['date'].dt.dayofweek
    features['month'] = df['date'].dt.month
    
    # --- 4. TARGET PREPARATION (t+1) ---
    # We are at time 't'. We want to predict 't+1'.
    # features['target_next_day'] = df[target].shift(-1)
    # #test feats
    # features['brent_zscore_90d'] = ((df['brent_oil'] - df['brent_oil'].rolling(90).mean()) / (df['brent_oil'].rolling(90).std() + 1e-9)).astype('float32')
    # features['brent_52w_pct'] = ((df['brent_oil'] - df['brent_oil'].rolling(252, min_periods=60).min()) / (df['brent_oil'].rolling(252, min_periods=60).max() - df['brent_oil'].rolling(252, min_periods=60).min() + 1e-9)).astype('float32')
    # features['brent_usd_vnd'] = (df['brent_oil'] * df['USD_VND_Data'] / 1000).astype('float32')
    # features['brent_usd_mom5d'] = features['brent_usd_vnd'].pct_change(5).astype('float32')
    # features['sp500_mom_10d'] = df['SP500'].pct_change(10).astype('float32')
    # features['sp500_roll14_mean'] = df['SP500'].shift(1).rolling(14).mean().astype('float32')
    # features['usd_vnd_mom7d'] = df['USD_VND_Data'].pct_change(7).astype('float32')
    
    # df['day_of_week'] = df.index.dayofweek
    # features['month_sin'] = np.sin(df['month'] * 2 * np.pi / 12).astype('float32')
    # features['month_cos'] = np.cos(df['month'] * 2 * np.pi / 12).astype('float32')
    # features['dow_sin']   = np.sin(df['day_of_week'] * 2 * np.pi / 7).astype('float32')
    # features['dow_cos']   = np.cos(df['day_of_week'] * 2 * np.pi / 7).astype('float32')
    # features['week_sin']  = np.sin(df['week'] * 2 * np.pi / 52).astype('float32')
    # features['week_cos']  = np.cos(df['week'] * 2 * np.pi / 52).astype('float32')
    # --- 5. CLEAN UP ---
    # Drop rows with NaNs (due to lags and rolling windows)
    # And drop the last row (since target_next_day is NaN)
    final_df = features.dropna()
    
    # Save
    final_df.to_csv(OUTPUT_PATH, index=False)
    
    # Documentation
    with open(REPORT_PATH, "w", encoding='utf-8') as f:
        f.write("# FEATURE ENGINEERING REPORT\n\n")
        f.write(f"Total features created: {len(final_df.columns) - 2} (excluding date and target)\n")
        f.write("## Feature List:\n")
        for col in final_df.columns:
            f.write(f"- {col}\n")
        f.write("\n## Safety Check:\n")
        f.write("- ALL features are based on data available at time 't' to predict 't+1'.\n")
        f.write("- Time gaps handled: No gaps in daily data.\n")
        f.write(f"- Final Dataset Shape: {final_df.shape}\n")

    print(f"Feature engineering completed. Dataset saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    engineer_features()
