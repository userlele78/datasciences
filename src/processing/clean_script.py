import pandas as pd
import numpy as np
import os

# Configuration
INPUT_PATH = r"f:\Project\DataSciences\merged_processed_data (1).csv"
OUTPUT_PATH = r"f:\Project\DataSciences\clean_data.csv"

def clean_data():
    # 1. Load data
    df = pd.read_csv(INPUT_PATH)
    
    # 2. Drop empty 'Unnamed' columns
    empty_cols = [col for col in df.columns if 'Unnamed' in col]
    df = df.drop(columns=empty_cols)
    
    # 3. Normalize Column Names (Fix encoding issues)
    rename_map = {
        'Date': 'date',
        'FEDFUNDS': 'fed_funds',
        'SP500': 'sp500',
        'VangVN_Buy Price': 'gold_buy',
        'VangVN_Sell Price': 'gold_sell',
        'Brent_Oil_Data': 'brent_oil',
        'Baltic_Dry_Index': 'baltic_dry',
        'USD_VND_Data': 'usd_vnd',
        'Oil_Rigs': 'oil_rigs',
        'Gas_Rigs': 'gas_rigs',
        'Misc_Rigs': 'misc_rigs',
        'D?u DO 0,05S-II': 'diesel_do',
        'X?ng E5 RON 92-II': 'gas_e5_ron92',
        'X?ng RON 95-III': 'gas_ron95'
    }
    df = df.rename(columns=rename_map)
    
    # 4. Convert Time & Sort
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # 5. Handle Missing Values
    # Forward fill for time-series (propagate last known value)
    df = df.ffill()
    # If there are still NaNs at the start, backfill
    df = df.bfill()
    
    # 6. Outlier Detection (Z-Score) - Just for reporting/analysis later
    # We won't remove them yet as they might be price jumps
    def detect_outliers_iqr(series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return ((series < lower_bound) | (series > upper_bound)).sum()

    outlier_counts = {col: detect_outliers_iqr(df[col]) for col in df.select_dtypes(include=[np.number]).columns}
    print("Outliers detected (IQR):", outlier_counts)

    # 7. CRITICAL: Alignment / Shifting
    # Our objective is y(t+1) = f(X(t))
    # In the raw data, row i has X(i) and y(i). 
    # To predict y(i+1) using X(i), we effectively use current features for future target.
    # We will create the target column 'target_next_day' by shifting the target variable
    # But for Step 2, we just ensure clean alignment. Feature engineering will handle specific lags.
    
    # Save clean data
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Cleaned data saved to {OUTPUT_PATH}")
    print(f"Final shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

if __name__ == "__main__":
    clean_data()
