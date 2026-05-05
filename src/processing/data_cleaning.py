import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def clean_data(input_path, output_path):
    df = pd.read_csv(input_path)
    
    # Pruning empty columns
    empty_cols = [col for col in df.columns if 'Unnamed' in col]
    df = df.drop(columns=empty_cols)
    
    # Normalize Names
    rename_map = {
        'Date': 'date', 'FEDFUNDS': 'fed_funds', 'SP500': 'sp500',
        'VangVN_Buy Price': 'gold_buy', 'VangVN_Sell Price': 'gold_sell',
        'Brent_Oil_Data': 'brent_oil', 'Baltic_Dry_Index': 'baltic_dry',
        'USD_VND_Data': 'usd_vnd', 'Oil_Rigs': 'oil_rigs',
        'Gas_Rigs': 'gas_rigs', 'Misc_Rigs': 'misc_rigs',
        'D?u DO 0,05S-II': 'diesel_do', 'X?ng E5 RON 92-II': 'gas_e5_ron92',
        'X?ng RON 95-III': 'gas_ron95'
    }
    df = df.rename(columns=rename_map)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').ffill().bfill()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    RAW_DATA = "f:/Project/DataSciences/data/raw/merged_data.csv"
    PROCESSED_DATA = "f:/Project/DataSciences/data/processed/clean_data.csv"
    clean_data(RAW_DATA, PROCESSED_DATA)
