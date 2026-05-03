import os
import json
import joblib
import pandas as pd
import sys

# Ensure utils module can be loaded correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error

from utils.helper_functions import ensure_datasets_exist
from utils.preprocess import preprocess_data

def get_model_instances():
    return {
        'Linear Regression': LinearRegression(),
        'KNN': KNeighborsRegressor(n_neighbors=5),
        'SVR': SVR(kernel='rbf', C=1000, gamma='scale'),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

def train_and_evaluate(device_type, df, output_dir='models'):
    X_train, X_test, y_train, y_test, scaler, le_dict = preprocess_data(df)
    instances = get_model_instances()
    metrics = []
    
    best_r2 = -float('inf')
    best_model = None
    best_model_name = ""

    for name, model in instances.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        
        metrics.append({
            'name': name,
            'r2': max(0.01, float(r2)),
            'accuracy': max(0.1, float(r2) * 100),
            'mae': float(mae)
        })
        
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_model_name = name

    metrics.sort(key=lambda x: x['r2'], reverse=True)
    
    # Save the artifacts
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(best_model, os.path.join(output_dir, f'{device_type}_model.pkl'))
    
    preprocessor = {'scaler': scaler, 'le_dict': le_dict}
    joblib.dump(preprocessor, os.path.join(output_dir, f'{device_type}_preprocessor.pkl'))
    
    return metrics, best_model_name

def run_training_pipeline():
    mobile_csv = os.path.join('datasets', 'mobile_prices_2026.csv')
    laptop_csv = os.path.join('datasets', 'laptop_prices_2026.csv')
    
    os.makedirs('datasets', exist_ok=True)
    ensure_datasets_exist(mobile_csv, laptop_csv)
    
    df_mobile = pd.read_csv(mobile_csv)
    df_laptop = pd.read_csv(laptop_csv)
    
    mobile_metrics, mobile_best = train_and_evaluate('mobile', df_mobile)
    laptop_metrics, laptop_best = train_and_evaluate('laptop', df_laptop)
    
    all_metrics = {
        'mobile': mobile_metrics,
        'laptop': laptop_metrics
    }
    
    with open(os.path.join('models', 'metrics.json'), 'w') as f:
        json.dump(all_metrics, f, indent=4)
        
    print(f"Training Complete! Best setup -> Mobile: {mobile_best}, Laptop: {laptop_best}")

if __name__ == "__main__":
    run_training_pipeline()
