import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
import shap
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.utils.metrics import calculate_metrics
from src.models.train_ensemble import FuelPriceEnsemble

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "processed" / "feature_dataset.csv"
WEEKLY_PATH = BASE_DIR / "data" / "raw" / "fuel_prices.csv"


def load_weekly_data(path):
    df_weekly = pd.read_csv(path)
    df_weekly.columns = df_weekly.columns.str.strip()

    date_col = None
    for col in df_weekly.columns:
        if col.lower() in ["date", "day", "time"]:
            date_col = col
            break

    if date_col is None:
        raise ValueError("Không tìm thấy cột Date trong weekly dataset")

    df_weekly[date_col] = pd.to_datetime(df_weekly[date_col])
    df_weekly = df_weekly.rename(columns={date_col: "Date"})
    df_weekly = df_weekly.sort_values("Date").reset_index(drop=True)
    return df_weekly


# =========================
# ## 🔥 SHAP FEATURE IMPORTANCE CHO RANDOM FOREST
# =========================
def compute_shap_rf(model, X_test, feature_names, output_dir=None):
    """
    Tính SHAP values cho Random Forest, bao gồm tác động âm và dương.
    
    Parameters:
        model: RandomForestRegressor đã được fit
        X_test: dữ liệu test (numpy array hoặc DataFrame)
        feature_names: danh sách tên feature
        output_dir: thư mục lưu plot (None = không lưu)
    
    Returns:
        shap_values: numpy array shape (n_samples, n_features)
        mean_pos: tác động dương trung bình theo feature
        mean_neg: tác động âm trung bình theo feature
    """
    print("\n📊 Đang tính SHAP values cho Random Forest...")

    # TreeExplainer nhanh hơn KernelExplainer cho tree-based models
    explainer = shap.TreeExplainer(model)

    # Giới hạn sample để tránh OOM (tối đa 500 mẫu)
    X_sample = X_test[:500] if len(X_test) > 500 else X_test
    if not isinstance(X_sample, pd.DataFrame):
        X_sample = pd.DataFrame(X_sample, columns=feature_names)

    shap_values = explainer.shap_values(X_sample)  # shape: (n_samples, n_features)

    # ---- Tách tác động âm và dương ----
    pos_shap = np.where(shap_values > 0, shap_values, 0)   # chỉ giữ dương
    neg_shap = np.where(shap_values < 0, shap_values, 0)   # chỉ giữ âm

    mean_abs  = np.abs(shap_values).mean(axis=0)            # tầm quan trọng tổng
    mean_pos  = pos_shap.mean(axis=0)                        # tác động dương TB
    mean_neg  = neg_shap.mean(axis=0)                        # tác động âm TB

    # ---- Tạo summary DataFrame ----
    shap_summary = pd.DataFrame({
        "feature":       feature_names,
        "mean_abs_shap": mean_abs,
        "mean_pos_shap": mean_pos,
        "mean_neg_shap": mean_neg,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    print("\n🔍 SHAP Feature Importance (RF) — Top 15:")
    print(shap_summary.head(15).to_string(index=False))

    # ---- Plot 1: Beeswarm / Summary plot (âm + dương rõ ràng) ----
    fig1, ax1 = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=feature_names,
        show=False,
        plot_type="dot",       # mỗi điểm = 1 mẫu, màu = giá trị feature
        max_display=20,
        plot_size=None,
        color_bar=True,
    )
    ax1 = plt.gca()
    ax1.set_title("SHAP Summary Plot — Random Forest\n(xanh = tác động âm, đỏ = tác động dương)", 
                  fontsize=12, pad=12)
    plt.tight_layout()

    # ---- Plot 2: Stacked bar âm/dương ----
    top_n = 15
    top_df = shap_summary.head(top_n).iloc[::-1]   # đảo để feature quan trọng nhất ở trên

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    bars_pos = ax2.barh(top_df["feature"], top_df["mean_pos_shap"],
                        color="#e74c3c", label="Tác động dương (+)", height=0.6)
    bars_neg = ax2.barh(top_df["feature"], top_df["mean_neg_shap"],
                        color="#2980b9", label="Tác động âm (−)",  height=0.6)

    ax2.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_xlabel("SHAP value trung bình", fontsize=11)
    ax2.set_title("SHAP Feature Importance — Tác động Âm / Dương\nRandom Forest", fontsize=12)
    ax2.legend(loc="lower right", fontsize=10)
    plt.tight_layout()

    # ---- Lưu file nếu có output_dir ----
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        path_beeswarm = output_dir / "shap_rf_beeswarm.png"
        path_bar      = output_dir / "shap_rf_signed_bar.png"
        path_csv      = output_dir / "shap_rf_summary.csv"

        fig1.savefig(path_beeswarm, dpi=150, bbox_inches="tight")
        fig2.savefig(path_bar,      dpi=150, bbox_inches="tight")
        shap_summary.to_csv(path_csv, index=False)

        print(f"\n✅ Đã lưu SHAP plots → {output_dir}")

    plt.show()

    return shap_values, mean_pos, mean_neg, shap_summary
# =========================


def run_comparison():
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])

    df_weekly = load_weekly_data(WEEKLY_PATH)
    event_mask_full = df['date'].isin(df_weekly['Date']).values

    X = df.drop(columns=['date', 'target_delta'])
    y = df['target_delta']

    start_idx = 2000
    step_size = 30

    models = {
        "XGBoost":           xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=42),
        "LightGBM":          lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42, verbose=-1),
        "RandomForest":      RandomForestRegressor(n_estimators=100, random_state=42),
        "Ensemble (RF+LGBM)": FuelPriceEnsemble()
    }

    results = {}
    rf_model_final   = None   # 🔥 lưu RF đã fit cuối cùng để tính SHAP
    rf_test_x_final  = None   # 🔥 lưu test set cuối cùng

    for name, model in models.items():
        print(f"Evaluating {name}...")
        all_actuals = []
        all_preds = []

        for i in range(start_idx, len(df), step_size):
            train_x, train_y = X.iloc[:i], y.iloc[:i]
            test_x,  test_y  = X.iloc[i:i+step_size], y.iloc[i:i+step_size]

            if len(test_x) == 0:
                break

            model.fit(train_x, train_y)
            pred_delta = model.predict(test_x)

            current_prices = test_x['price_lag_0'].values
            all_preds.extend(current_prices + pred_delta)
            all_actuals.extend(current_prices + test_y.values)

            # 🔥 Ghi nhớ RF và test set của bước cuối để dùng SHAP
            if name == "RandomForest":
                rf_model_final  = model
                rf_test_x_final = test_x.copy()

        mask = event_mask_full[start_idx : start_idx + len(all_actuals)]
        results[name] = calculate_metrics(
            np.array(all_actuals),
            np.array(all_preds),
            mask=mask
        )

    # =========================
    ## 🔥 SHAP CHO RANDOM FOREST (sau khi vòng lặp kết thúc)
    # =========================
    # if rf_model_final is not None and rf_test_x_final is not None:
    #     shap_output_dir = BASE_DIR / "logs" / "shap"
    #     shap_values, mean_pos, mean_neg, shap_summary = compute_shap_rf(
    #         model        = rf_model_final,
    #         X_test       = rf_test_x_final,
    #         feature_names= list(X.columns),
    #         output_dir   = shap_output_dir
    #     )
    # =========================

    # =========================
    # SARIMAX
    # =========================
    print("Evaluating SARIMAX...")
    ts_data  = df['price_lag_0']
    exo_data = df[['brent_lag_0']]

    sarimax_actuals = []
    sarimax_preds   = []

    for i in range(start_idx, len(df), step_size):
        train_y   = ts_data.iloc[:i]
        train_exo = exo_data.iloc[:i]
        test_y    = ts_data.iloc[i:i+step_size]
        test_exo  = exo_data.iloc[i:i+step_size]

        if len(test_y) == 0:
            break

        try:
            model_sm = SARIMAX(train_y, exog=train_exo, order=(1, 1, 1))
            res      = model_sm.fit(disp=False)
            preds    = res.forecast(steps=len(test_y), exog=test_exo)
            sarimax_preds.extend(preds)
            sarimax_actuals.extend(test_y.values)
        except:
            continue

    mask_sarimax = event_mask_full[start_idx : start_idx + len(sarimax_actuals)]
    results["SARIMAX"] = calculate_metrics(
        np.array(sarimax_actuals),
        np.array(sarimax_preds),
        mask=mask_sarimax
    )

    # =========================
    # REPORT
    # =========================
    print("\n" + "=" * 30)
    print("FINAL COMPARISON REPORT (WEEKLY DA)")
    print("=" * 30)

    report_df = pd.DataFrame(results).T
    print(report_df)

    OUTPUT_PATH = BASE_DIR / "logs" / "model_comparison_final.csv"
    report_df.to_csv(OUTPUT_PATH)
    
    for name, model in models.items():
        print(f"Evaluating {name}...")
        all_actuals = []
        all_preds = []

        for i in range(start_idx, len(df), step_size):
            train_x, train_y = X.iloc[:i], y.iloc[:i]
            test_x,  test_y  = X.iloc[i:i+step_size], y.iloc[i:i+step_size]

            if len(test_x) == 0:
                break

            model.fit(train_x, train_y)
            pred_delta = model.predict(test_x)

            current_prices = test_x['price_lag_0'].values
            all_preds.extend(current_prices + pred_delta)
            all_actuals.extend(current_prices + test_y.values)

            # 🔥 Ghi nhớ RF và test set của bước cuối để dùng SHAP
            if name == "RandomForest":
                rf_model_final  = model
                rf_test_x_final = test_x.copy()

        mask = event_mask_full[start_idx : start_idx + len(all_actuals)]
        results[name] = calculate_metrics(
            np.array(all_actuals),
            np.array(all_preds),
            mask=mask
        )
        if rf_model_final is not None and rf_test_x_final is not None:
            shap_output_dir = BASE_DIR / "logs" / "shap"
            shap_values, mean_pos, mean_neg, shap_summary = compute_shap_rf(
                model        = rf_model_final,
                X_test       = rf_test_x_final,
                feature_names= list(X.columns),
                output_dir   = shap_output_dir
            )


if __name__ == "__main__":
    run_comparison()