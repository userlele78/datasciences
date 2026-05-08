import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


# =========================
# Event-based Directional Accuracy
# =========================
def directional_accuracy_event(y_true, y_pred, mask):
    y_true_f = y_true[mask]
    y_pred_f = y_pred[mask]
    
    if len(y_true_f) < 2:
        return np.nan
    
    y_true_diff = np.diff(y_true_f)
    y_pred_diff = np.diff(y_pred_f)
    
    direction_true = np.sign(y_true_diff)
    direction_pred = np.sign(y_pred_diff)
    
    return np.mean(direction_true == direction_pred)


# =========================
# Full metrics
# =========================
def calculate_metrics(y_true, y_pred, mask=None, debug=False):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    if debug and mask is not None:
        print("=== DEBUG ===")
        print(f"Tổng số điểm          : {len(y_true)}")
        print(f"Số change points      : {mask.sum()}")
        print(f"% change points       : {mask.mean():.4f}")

        pred_diff = np.diff(y_pred, prepend=y_pred[0])
        print(f"Số ngày pred thay đổi : {(pred_diff != 0).sum()}")
        print(f"Pred diff == 0 toàn bộ: {(pred_diff == 0).all()}")

        y_true_cp = y_true[mask]
        y_pred_cp = y_pred[mask]
        print(f"\nTại change points:")
        print(f"  y_true sample        : {y_true_cp[:5]}")
        print(f"  y_pred sample        : {y_pred_cp[:5]}")
        print(f"  y_true == y_pred     : {(y_true_cp == y_pred_cp).mean():.4f}")
        print(f"  diff y_true          : {np.diff(y_true_cp[:6])}")
        print(f"  diff y_pred          : {np.diff(y_pred_cp[:6])}")

    if mask is not None:
        da = directional_accuracy_event(y_true, y_pred, mask)
    else:
        y_true_diff = np.diff(y_true)
        y_pred_diff = np.diff(y_pred)
        direction_true = np.sign(y_true_diff)
        direction_pred = np.sign(y_pred_diff)
        da = np.mean(direction_true == direction_pred)

    return {
        "RMSE"                : rmse,
        "MAE"                 : mae,
        "MAPE"                : mape,
        "Directional_Accuracy": da
    }