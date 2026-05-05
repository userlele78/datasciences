import pandas as pd
import numpy as np
import os

# Configuration
DATA_PATH = r"f:\Project\DataSciences\merged_processed_data (1).csv"
OUTPUT_REPORT = r"f:\Project\DataSciences\data_audit_report.txt"

def perform_audit(df):
    report = []
    report.append("=== DATA AUDIT REPORT ===")
    
    # 1. Basic Info
    report.append(f"Shape: {df.shape}")
    report.append("\n--- Data Types ---")
    report.append(df.dtypes.to_string())
    
    # 2. Identify potential Time column
    date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    report.append(f"\nPotential Time Columns: {date_cols}")
    
    # 3. Missing Values
    missing = df.isnull().sum()
    report.append("\n--- Missing Values ---")
    report.append(missing[missing > 0].to_string() if not missing[missing > 0].empty else "No missing values")
    
    # 4. Duplicates
    duplicates = df.duplicated().sum()
    report.append(f"\nDuplicate Rows: {duplicates}")
    
    # 5. Time Continuity & Frequency (Assuming first date-like column is the time index)
    if date_cols:
        time_col = date_cols[0]
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(time_col)
        
        time_diffs = df[time_col].diff().dropna()
        freq_counts = time_diffs.value_counts()
        
        report.append(f"\n--- Time Analysis (Target Column: {time_col}) ---")
        report.append(f"Min Time: {df[time_col].min()}")
        report.append(f"Max Time: {df[time_col].max()}")
        report.append(f"Observed Frequencies:\n{freq_counts.head().to_string()}")
        
        # Detect gaps
        expected_freq = freq_counts.idxmax()
        gaps = time_diffs[time_diffs != expected_freq]
        report.append(f"\nIrregularities/Gaps detected: {len(gaps)}")
        if len(gaps) > 0:
            report.append(f"Example gap sizes: {gaps.unique()[:5]}")
            
    # 6. Target Variable Identification
    # Usually the last column or specific names like 'price', 'target', 'close'
    potential_targets = [col for col in df.columns if col.lower() in ['price', 'target', 'close', 'value']]
    report.append(f"\nPotential Target Variables: {potential_targets}")

    # Write report
    with open(OUTPUT_REPORT, "w", encoding='utf-8') as f:
        f.write("\n".join(report))
    
    print("Audit completed. Report saved to:", OUTPUT_REPORT)

if __name__ == "__main__":
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        perform_audit(df)
        print("Data loaded successfully.")
    else:
        print(f"Error: File {DATA_PATH} not found.")
