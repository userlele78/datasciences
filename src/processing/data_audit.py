import pandas as pd
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def run_audit(input_path, output_report):
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return
        
    df = pd.read_csv(input_path)
    report = [
        "=== DATA AUDIT REPORT ===",
        f"Shape: {df.shape}",
        "\n--- Data Types ---",
        df.dtypes.to_string(),
        f"\nDuplicate Rows: {df.duplicated().sum()}",
        "\n--- Missing Values ---",
        df.isnull().sum()[df.isnull().sum() > 0].to_string() or "No missing values"
    ]
    
    os.makedirs(os.path.dirname(output_report), exist_ok=True)
    with open(output_report, "w", encoding='utf-8') as f:
        f.write("\n".join(report))
    print(f"Audit report saved to {output_report}")

if __name__ == "__main__":
    RAW_DATA = "f:/Project/DataSciences/data/raw/merged_data.csv"
    REPORT = "f:/Project/DataSciences/logs/audit_report.txt"
    run_audit(RAW_DATA, REPORT)
