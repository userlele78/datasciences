import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

def calculate_metrics(y_true, y_pred):
    """Calculates regression metrics including directional accuracy."""
    
    # Basic metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    # Directional Accuracy
    y_true_diff = np.diff(y_true)
    y_pred_diff = y_pred[1:] - y_true[:-1]   # so với actual trước đó
    
    direction_true = np.sign(y_true_diff)
    direction_pred = np.sign(y_pred_diff)
    
    directional_accuracy = np.mean(direction_true == direction_pred)
    
    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "Directional_Accuracy": directional_accuracy
    }