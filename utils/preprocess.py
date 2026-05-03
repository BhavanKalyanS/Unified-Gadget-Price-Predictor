from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

def preprocess_data(df, target_col='Price'):
    """Prepares data by splitting and encoding."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    le_dict = {}
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        le_dict[col] = le
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, le_dict

def encode_input(input_dict, le_dict, scaler):
    """Encodes and scales user input for prediction."""
    df_input = pd.DataFrame([input_dict])
    for col, le in le_dict.items():
        if col in df_input.columns:
            try:
                df_input[col] = le.transform(df_input[col])
            except ValueError:
                df_input[col] = 0
                
    X_scaled = scaler.transform(df_input)
    return X_scaled
