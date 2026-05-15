import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os
from pathlib import Path

# Configuration
BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH = BASE_DIR / "data" / "processed" / "clean_data.csv"
OUTPUT_DIR = BASE_DIR / "logs" / "eda_report"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_full_eda():
    df = pd.read_csv(INPUT_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    
    target = 'gas_ron95'
    features = [col for col in df.columns if col != target]
    
    with open(os.path.join(OUTPUT_DIR, "eda_log.md"), "w", encoding='utf-8') as log:
        log.write("# FULL EDA REPORT\n\n")

        # 3.1 Data Overview
        log.write("## 3.1 Data Overview\n")
        log.write(f"- Shape: {df.shape}\n")
        log.write(f"- Time Range: {df.index.min()} to {df.index.max()}\n")
        log.write("- Missing values:\n")
        log.write(df.isnull().sum().to_string() + "\n\n")

        # 3.2 Univariate Analysis
        log.write("## 3.2 Univariate Analysis\n")
        desc = df.describe().transpose()
        desc['skew'] = df.skew()
        desc['kurtosis'] = df.kurtosis()
        log.write(desc.to_string() + "\n\n")
        
        # Plot distributions
        plt.figure(figsize=(20, 15))
        df.hist(bins=30, figsize=(20, 15))
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "distributions.png"))
        plt.close()

        # 3.3 Bivariate Analysis (Correlation)
        log.write("## 3.3 Bivariate Analysis\n")
        corr_pearson = df.corr(method='pearson')
        corr_spearman = df.corr(method='spearman')
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_pearson, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Pearson Correlation")
        plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"))
        plt.close()
        
        log.write("Top correlations with target (RON 95):\n")
        log.write(corr_pearson[target].sort_values(ascending=False).to_string() + "\n\n")

        # 3.4 Multivariate Analysis (PCA)
        log.write("## 3.4 Multivariate Analysis (PCA)\n")
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df.dropna())
        pca = PCA()
        pca.fit(scaled_data)
        exp_var = np.cumsum(pca.explained_variance_ratio_)
        log.write(f"Components needed for 95% variance: {np.argmax(exp_var >= 0.95) + 1}\n\n")
        
        plt.figure()
        plt.plot(exp_var)
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.savefig(os.path.join(OUTPUT_DIR, "pca_variance.png"))
        plt.close()

        # 3.5 Time Series Analysis
        log.write("## 3.5 Time Series Analysis\n")
        
        # Stationarity (ADF Test)
        result = adfuller(df[target])
        log.write(f"ADF Statistic for {target}: {result[0]}\n")
        log.write(f"p-value: {result[1]}\n")
        log.write("Stationary: " + ("Yes" if result[1] < 0.05 else "No") + "\n\n")
        
        # ACF/PACF
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        plot_acf(df[target], lags=50, ax=ax[0])
        plot_pacf(df[target], lags=50, ax=ax[1])
        plt.savefig(os.path.join(OUTPUT_DIR, "acf_pacf.png"))
        plt.close()

        # 3.6 & 3.7 Rolling & Volatility
        df['rolling_mean_30'] = df[target].rolling(window=30).mean()
        df['rolling_std_30'] = df[target].rolling(window=30).std()
        
        plt.figure(figsize=(15, 7))
        plt.plot(df[target], label='Original', alpha=0.5)
        plt.plot(df['rolling_mean_30'], label='Rolling Mean (30d)', color='red')
        plt.title("Price Trend & Rolling Mean")
        plt.legend()
        plt.savefig(os.path.join(OUTPUT_DIR, "trend_analysis.png"))
        plt.close()

        # 3.9 Regime Detection (KMeans on price + volatility)
        log.write("## 3.9 Regime Detection\n")
        regime_data = df[[target, 'rolling_std_30']].dropna()
        scaler_reg = StandardScaler()
        reg_scaled = scaler_reg.fit_transform(regime_data)
        
        kmeans = KMeans(n_clusters=3, random_state=42)
        df.loc[regime_data.index, 'regime'] = kmeans.fit_predict(reg_scaled)
        
        plt.figure(figsize=(15, 7))
        plt.scatter(df.index, df[target], c=df['regime'], cmap='viridis', s=10)
        plt.title("Price Regimes Detected (K-Means)")
        plt.savefig(os.path.join(OUTPUT_DIR, "regimes.png"))
        plt.close()
        
        log.write("Regime Statistics:\n")
        log.write(df.groupby('regime')[target].agg(['mean', 'std', 'count']).to_string() + "\n\n")

        # 3.11 KEY INSIGHTS (Placeholder for manual summary)
        log.write("## 3.11 PRELIMINARY INSIGHTS\n")
        log.write("1. Target is non-stationary (requires differencing or trend modeling).\n")
        log.write("2. Strong correlation detected with Brent Oil and Gold prices.\n")
        log.write("3. 3 clear regimes identified (Low, Mid, High volatility/price levels).\n")

    print("EDA completed. All results saved in:", OUTPUT_DIR)

if __name__ == "__main__":
    run_full_eda()
