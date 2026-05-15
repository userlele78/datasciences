import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
import shap
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.utils.metrics import calculate_metrics
from src.models.train_ensemble import FuelPriceEnsemble

BASE_DIR    = Path(__file__).resolve().parents[2]
DATA_PATH   = BASE_DIR / "data" / "processed" / "feature_dataset.csv"
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
# SHAP FEATURE IMPORTANCE CHO RANDOM FOREST
# =========================
def compute_shap_rf(model, X_test, feature_names, output_dir=None):
    print("\n📊 Đang tính SHAP values cho Random Forest...")

    explainer = shap.TreeExplainer(model)

    X_sample = X_test[:500] if len(X_test) > 500 else X_test
    if not isinstance(X_sample, pd.DataFrame):
        X_sample = pd.DataFrame(X_sample, columns=feature_names)

    shap_values = explainer.shap_values(X_sample)

    pos_shap = np.where(shap_values > 0, shap_values, 0)
    neg_shap = np.where(shap_values < 0, shap_values, 0)

    mean_abs = np.abs(shap_values).mean(axis=0)
    mean_pos = pos_shap.mean(axis=0)
    mean_neg = neg_shap.mean(axis=0)

    shap_summary = pd.DataFrame({
        "feature":       feature_names,
        "mean_abs_shap": mean_abs,
        "mean_pos_shap": mean_pos,
        "mean_neg_shap": mean_neg,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    print("\n🔍 SHAP Feature Importance (RF) — Top 15:")
    print(shap_summary.head(15).to_string(index=False))

    fig1, ax1 = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=feature_names,
        show=False,
        plot_type="dot",
        max_display=20,
        plot_size=None,
        color_bar=True,
    )
    ax1 = plt.gca()
    ax1.set_title(
        "SHAP Summary Plot — Random Forest\n(xanh = tác động âm, đỏ = tác động dương)",
        fontsize=12, pad=12,
    )
    plt.tight_layout()

    top_n  = 15
    top_df = shap_summary.head(top_n).iloc[::-1]

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.barh(top_df["feature"], top_df["mean_pos_shap"],
             color="#e74c3c", label="Tác động dương (+)", height=0.6)
    ax2.barh(top_df["feature"], top_df["mean_neg_shap"],
             color="#2980b9", label="Tác động âm (−)", height=0.6)
    ax2.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_xlabel("SHAP value trung bình", fontsize=11)
    ax2.set_title("SHAP Feature Importance — Tác động Âm / Dương\nRandom Forest", fontsize=12)
    ax2.legend(loc="lower right", fontsize=10)
    plt.tight_layout()

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig1.savefig(output_dir / "shap_rf_beeswarm.png",   dpi=150, bbox_inches="tight")
        fig2.savefig(output_dir / "shap_rf_signed_bar.png", dpi=150, bbox_inches="tight")
        shap_summary.to_csv(output_dir / "shap_rf_summary.csv", index=False)
        print(f"\n✅ Đã lưu SHAP plots → {output_dir}")

    plt.show()
    return shap_values, mean_pos, mean_neg, shap_summary


# =========================
# PLOT ACTUAL VS PREDICTED  (date-aligned)
# =========================
def plot_actual_vs_predicted(dates, actuals, all_preds_dict, all_dates_dict,
                              actuals_dict, results, output_dir):
    """
    Align tất cả predictions theo timestamp thực (reindex) trước khi vẽ,
    tránh lệch phase khi các model có số bước walk-forward khác nhau.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    color_map = {
        "XGBoost":            "#e74c3c",
        "LightGBM":           "#2ecc71",
        "RandomForest":       "#9b59b6",
        "Ensemble (RF+LGBM)": "#f39c12",
        "SARIMAX":            "#1abc9c",
    }

    # ── Build aligned DataFrame ──────────────────────────────────────────
    ref_dates = pd.to_datetime(dates)
    plot_df   = pd.DataFrame({"actual": actuals}, index=ref_dates)
    plot_df.index.name = "date"

    for name, preds in all_preds_dict.items():
        model_dates = pd.to_datetime(all_dates_dict[name])
        s = pd.Series(preds, index=model_dates, name=name)
        # reindex theo ref_dates: NaN nếu model thiếu ngày đó
        plot_df[name] = s.reindex(ref_dates)

    # --- Tìm mô hình tốt nhất theo RMSE ---
    best_name = min(results, key=lambda m: results[m].get("rmse", float("inf")))

    # ── Plot 1: Best model only ──────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(14, 4))
    ax1.plot(plot_df.index, plot_df["actual"],
             color="#2c3e50", linewidth=1.5, label="Giá thật")
    ax1.plot(plot_df.index, plot_df[best_name],
             color=color_map.get(best_name, "#e74c3c"),
             linewidth=1.2, linestyle="--", label=f"{best_name} (best)")

    ax1.set_title(f"Giá Thật vs Mô Hình Tốt Nhất — {best_name}",
                  fontsize=12, fontweight="bold")
    ax1.set_ylabel("Giá nhiên liệu")
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig1.autofmt_xdate(rotation=30)
    plt.tight_layout()

    path1 = output_dir / "actual_vs_best_model.png"
    fig1.savefig(path1, dpi=150, bbox_inches="tight")
    print(f"✅ Đã lưu → {path1}")
    plt.show()

    # ── Plot 2: All models ───────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(14, 4))
    ax2.plot(plot_df.index, plot_df["actual"],
             color="#2c3e50", linewidth=1.8, label="Giá thật", zorder=5)

    for name in all_preds_dict:
        if name in plot_df.columns:
            ax2.plot(plot_df.index, plot_df[name],
                     color=color_map.get(name, "#3498db"),
                     linewidth=1.0, linestyle="--", alpha=0.75, label=name)

    ax2.set_title("Giá Thật vs Tất cả Mô Hình", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Giá nhiên liệu")
    ax2.legend(fontsize=9, framealpha=0.7)
    ax2.grid(axis="y", linestyle="--", alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig2.autofmt_xdate(rotation=30)
    plt.tight_layout()

    path2 = output_dir / "actual_vs_all_models.png"
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    print(f"✅ Đã lưu → {path2}")
    plt.show()


# =========================
# MAIN
# =========================
def run_comparison():
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])

    df_weekly       = load_weekly_data(WEEKLY_PATH)
    event_mask_full = df['date'].isin(df_weekly['Date']).values

    X          = df.drop(columns=['date', 'target_delta'])
    y          = df['target_delta']
    dates_full = df['date'].values  # numpy datetime64

    start_idx = 2000
    step_size = 30

    models = {
        "XGBoost":            xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=42),
        "LightGBM":           lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42, verbose=-1),
        "RandomForest":       RandomForestRegressor(n_estimators=100, random_state=42),
        "Ensemble (RF+LGBM)": FuelPriceEnsemble(),
    }

    results        = {}
    all_preds_dict = {}   # predictions theo mô hình
    all_dates_dict = {}   # dates theo mô hình  ← FIX: lưu riêng để align
    actuals_dict   = {}   # actuals theo mô hình ← FIX: lưu riêng
    actuals_shared = None
    dates_shared   = None

    rf_model_final  = None
    rf_test_x_final = None

    # =========================
    # ML MODELS — walk-forward
    # =========================
    for name, model in models.items():
        print(f"Evaluating {name}...")
        all_actuals = []
        all_preds   = []
        all_dates   = []

        for i in range(start_idx, len(df), step_size):
            train_x, train_y = X.iloc[:i],             y.iloc[:i]
            test_x,  test_y  = X.iloc[i:i+step_size],  y.iloc[i:i+step_size]

            if len(test_x) == 0:
                break

            model.fit(train_x, train_y)
            pred_delta = model.predict(test_x)

            current_prices = test_x['price_lag_0'].values
            all_preds.extend(current_prices + pred_delta)
            all_actuals.extend(current_prices + test_y.values)
            all_dates.extend(dates_full[i:i+step_size])

            if name == "RandomForest":
                rf_model_final  = model
                rf_test_x_final = test_x.copy()

        mask = event_mask_full[start_idx : start_idx + len(all_actuals)]
        results[name] = calculate_metrics(
            np.array(all_actuals),
            np.array(all_preds),
            mask=mask,
        )

        # FIX: lưu đầy đủ per-model thay vì chỉ lưu lần đầu
        all_preds_dict[name] = np.array(all_preds)
        all_dates_dict[name] = np.array(all_dates, dtype="datetime64[ns]")
        actuals_dict[name]   = np.array(all_actuals)

        if actuals_shared is None:
            actuals_shared = np.array(all_actuals)
            dates_shared   = np.array(all_dates, dtype="datetime64[ns]")

    # =========================
    # SHAP — Random Forest
    # =========================
    if rf_model_final is not None and rf_test_x_final is not None:
        shap_values, mean_pos, mean_neg, shap_summary = compute_shap_rf(
            model         = rf_model_final,
            X_test        = rf_test_x_final,
            feature_names = list(X.columns),
            output_dir    = BASE_DIR / "logs" / "shap",
        )

    # =========================
    # SARIMAX
    # =========================
    print("Evaluating SARIMAX...")
    ts_data  = df['price_lag_0']
    exo_data = df[['brent_lag_0']]

    sarimax_actuals = []
    sarimax_preds   = []
    sarimax_dates   = []

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
            sarimax_dates.extend(dates_full[i:i+step_size])
        except Exception:
            continue  # skip step nếu SARIMAX fail — dates vẫn align đúng nhờ reindex

    mask_sarimax = event_mask_full[start_idx : start_idx + len(sarimax_actuals)]
    results["SARIMAX"] = calculate_metrics(
        np.array(sarimax_actuals),
        np.array(sarimax_preds),
        mask=mask_sarimax,
    )

    # FIX: lưu dates riêng cho SARIMAX
    all_preds_dict["SARIMAX"] = np.array(sarimax_preds)
    all_dates_dict["SARIMAX"] = np.array(sarimax_dates, dtype="datetime64[ns]")
    actuals_dict["SARIMAX"]   = np.array(sarimax_actuals)

    # =========================
    # REPORT
    # =========================
    print("\n" + "=" * 30)
    print("FINAL COMPARISON REPORT (WEEKLY DA)")
    print("=" * 30)

    report_df = pd.DataFrame(results).T
    print(report_df)

    LOG_DIR = BASE_DIR / "logs"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(LOG_DIR / "model_comparison_final.csv")
    print(f"\n✅ Đã lưu báo cáo CSV → {LOG_DIR / 'model_comparison_final.csv'}")

    # =========================
    # PLOT ACTUAL VS PREDICTED
    # =========================
    if actuals_shared is not None and dates_shared is not None:
        plot_actual_vs_predicted(
            dates          = dates_shared,
            actuals        = actuals_shared,
            all_preds_dict = all_preds_dict,   # FIX: truyền dict đầy đủ
            all_dates_dict = all_dates_dict,   # FIX: truyền dates per-model
            actuals_dict   = actuals_dict,
            results        = results,
            output_dir     = LOG_DIR,
        )


if __name__ == "__main__":
    run_comparison()