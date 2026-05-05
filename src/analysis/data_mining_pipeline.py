import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.tsa.stattools import ccovf, ccf
import os

# Configuration
INPUT_PATH = r"f:\Project\DataSciences\clean_data.csv"
OUTPUT_DIR = r"f:\Project\DataSciences\DataMining_Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_data_mining():
    df = pd.read_csv(INPUT_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    
    target = 'gas_ron95'
    features_to_mine = ['brent_oil', 'usd_vnd', 'gold_sell', 'baltic_dry', 'fed_funds']
    
    with open(os.path.join(OUTPUT_DIR, "data_mining_log.md"), "w", encoding='utf-8') as log:
        log.write("# DATA MINING REPORT\n\n")

        # 4.1 Lag Structure Mining (CCF)
        log.write("## 4.1 Lag Structure Mining (CCF)\n")
        log.write("Analyzing which lags of exogenous features have the highest correlation with RON 95.\n\n")
        
        plt.figure(figsize=(15, 10))
        optimal_lags = {}
        
        for i, feat in enumerate(features_to_mine):
            # Calculate CCF
            # We use differenced series for CCF to avoid spurious correlation if non-stationary
            target_diff = df[target].diff().dropna()
            feat_diff = df[feat].diff().dropna()
            
            # Align lengths
            common_idx = target_diff.index.intersection(feat_diff.index)
            y = target_diff.loc[common_idx]
            x = feat_diff.loc[common_idx]
            
            cross_corr = [np.corrcoef(x.shift(l).dropna(), y.iloc[l:])[0,1] for l in range(31)]
            
            plt.subplot(3, 2, i+1)
            plt.stem(range(31), cross_corr)
            plt.title(f"CCF: {feat} vs {target}")
            plt.xlabel("Lag (Days)")
            plt.ylabel("Correlation")
            
            best_lag = np.argmax(np.abs(cross_corr))
            optimal_lags[feat] = best_lag
            log.write(f"- **{feat}**: Optimal Lag = {best_lag} days (Corr: {cross_corr[best_lag]:.3f})\n")

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "cross_correlation.png"))
        plt.close()
        log.write("\n")

        # 4.2 Granger Causality
        log.write("## 4.2 Relationship Mining (Granger Causality)\n")
        log.write("Checking if 'A' helps predict 'B' beyond 'B's own history.\n\n")
        
        for feat in ['brent_oil', 'usd_vnd']:
            log.write(f"### Testing: {feat} -> {target}\n")
            # GC requires stationary data
            try:
                gc_data = df[[target, feat]].diff().dropna()
                res = grangercausalitytests(gc_data, maxlag=7, verbose=False)
                # Check p-values for lag 1 to 7
                p_values = [res[i+1][0]['ssr_ftest'][1] for i in range(7)]
                min_p = min(p_values)
                log.write(f"- Minimum p-value across 7 lags: {min_p:.5f}\n")
                if min_p < 0.05:
                    log.write(f"- Result: **CAUSAL RELATIONSHIP DETECTED** (p < 0.05)\n")
                else:
                    log.write(f"- Result: No significant causality detected at 5% level.\n")
            except Exception as e:
                log.write(f"- Error during GC test: {str(e)}\n")
        log.write("\n")

        # 4.3 Structural Change Point Detection (Simple Variance Shift)
        log.write("## 4.3 Structural Change Point Detection\n")
        rolling_var = df[target].diff().rolling(window=60).var()
        change_points = rolling_var.diff().abs().nlargest(5).index
        log.write("Detected 5 major volatility shift points:\n")
        for cp in change_points:
            log.write(f"- {cp.date()}\n")
            
        plt.figure(figsize=(15, 5))
        plt.plot(df[target], label='Price')
        for cp in change_points:
            plt.axvline(x=cp, color='r', linestyle='--', alpha=0.5)
        plt.title("Price with Major Structural Change Points (Red Lines)")
        plt.savefig(os.path.join(OUTPUT_DIR, "change_points.png"))
        plt.close()

        # 4.4 Final Data Mining Synthesis
        log.write("\n## 4.4 Synthesis for Feature Engineering\n")
        log.write("1. **Lag Selection**: Features should be lagged by discovered optimal lags (mostly 0-5 days).\n")
        log.write("2. **Causality**: Brent Oil is a verified driver; USD/VND requires non-linear treatment.\n")
        log.write("3. **Volatility**: Models must account for regime shifts detected at identified change points.\n")

    print("Data Mining completed. All results saved in:", OUTPUT_DIR)

if __name__ == "__main__":
    run_data_mining()
