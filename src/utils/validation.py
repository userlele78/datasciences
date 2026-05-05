import numpy as np

def walk_forward_validation(df, model_func, start_idx, step_size):
    """
    Production-grade Walk-forward validation engine.
    """
    predictions = []
    actuals = []
    
    X = df.drop(columns=['date', 'target_next_day'])
    y = df['target_next_day']
    
    for i in range(start_idx, len(df), step_size):
        train_x = X.iloc[:i]
        train_y = y.iloc[:i]
        
        test_x = X.iloc[i:i+step_size]
        test_y = y.iloc[i:i+step_size]
        
        if len(test_x) == 0:
            break
            
        y_pred = model_func(train_x, train_y, test_x)
        predictions.extend(y_pred)
        actuals.extend(test_y.values)
        
    return np.array(actuals), np.array(predictions)
