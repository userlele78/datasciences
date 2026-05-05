import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# Paths
DATA_PATH = "f:/Project/DataSciences/data/processed/feature_dataset.csv"
MODEL_PATH = "f:/Project/DataSciences/models/final_rf_model.pkl"
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

def train_and_save():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=['date', 'target_delta'])
    y = df['target_delta']
    
    print("Training Final RandomForest Model on all data...")
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    
    joblib.dump(model, MODEL_PATH)
    print(f"Final Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_and_save()
