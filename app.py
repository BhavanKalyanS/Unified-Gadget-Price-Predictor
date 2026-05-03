import os
import json
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load Pre-trained artifacts into memory
MODEL_DIR = 'models'

mobile_model = None
laptop_model = None
mobile_preprocessor = None
laptop_preprocessor = None
metrics_data = {}

def load_artifacts():
    global mobile_model, laptop_model, mobile_preprocessor, laptop_preprocessor, metrics_data
    try:
        mobile_model = joblib.load(os.path.join(MODEL_DIR, 'mobile_model.pkl'))
        laptop_model = joblib.load(os.path.join(MODEL_DIR, 'laptop_model.pkl'))
        mobile_preprocessor = joblib.load(os.path.join(MODEL_DIR, 'mobile_preprocessor.pkl'))
        laptop_preprocessor = joblib.load(os.path.join(MODEL_DIR, 'laptop_preprocessor.pkl'))
        
        with open(os.path.join(MODEL_DIR, 'metrics.json'), 'r') as f:
            metrics_data = json.load(f)
        print("Models successfully loaded from disk.")
    except Exception as e:
        print(f"Error loading models: {e}. Ensure you have run python utils/train_models.py first.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    device = request.args.get('device', 'mobile')
    device_metrics = metrics_data.get(device, [])
    
    if not device_metrics:
        return jsonify(device_metrics)
        
    # User's frontend expects summary stats
    top_model = device_metrics[0]
    return jsonify({
        'total_predictions': 1284, # Placeholder for demo
        'accuracy': top_model['r2'], # frontend multiplies by 100
        'mae': top_model['mae'],
        'models': device_metrics # Keep original list for chart
    })

@app.route('/api/retrain', methods=['POST'])
def train():
    try:
        from utils.train_models import run_training_pipeline
        run_training_pipeline()
        load_artifacts() # Refresh memory
        return jsonify({'message': 'Models retrained successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    # Support both 'device' and 'device_type'
    device = data.get('device_type') or data.get('device') or 'mobile'
    
    try:
        print(f"DEBUG: Input Data: {data}")
        # Case-insensitive data getter to be robust
        def get_val(key):
            val = data.get(key)
            if val is None:
                val = data.get(key.lower())
            if val is None:
                val = data.get(key.replace('_', '').lower())
            if val is None:
                val = data.get(key.upper())
            return val

        if device == 'mobile':
            model = mobile_model
            preproc = mobile_preprocessor
            input_dict = {
                'RAM': float(get_val('RAM')),
                'Storage': float(get_val('Storage')),
                'Battery': float(get_val('Battery')),
                'Camera': float(get_val('Camera')),
                'RefreshRate': float(get_val('RefreshRate'))
            }
        else:
            model = laptop_model
            preproc = laptop_preprocessor
            input_dict = {
                'RAM': float(get_val('RAM')),
                'Storage': float(get_val('Storage')),
                'GPU': get_val('GPU'),
                'Screen': float(get_val('Screen')),
                'Processor': get_val('Processor')
            }
        
        print(f"DEBUG: input_dict: {input_dict}")
            
        df_input = pd.DataFrame([input_dict])
        
        le_dict = preproc['le_dict']
        scaler = preproc['scaler']
        
        for col, le in le_dict.items():
            if col in df_input.columns:
                try:
                    df_input[col] = le.transform(df_input[col])
                except ValueError:
                    df_input[col] = 0 
                    
        X_scaled = scaler.transform(df_input)
        
        prediction = model.predict(X_scaled)[0]
        best_metric = metrics_data[device][0]
        
        # Determine Segment
        price = round(prediction, 0)
        segment = "Premium"
        if price < 50000: segment = "Budget"
        elif price < 100000: segment = "Mid-Range"
        elif price > 200000: segment = "Ultra Flagship"

        return jsonify({
            'predicted_price': price,
            'confidence': best_metric['r2'],
            'model': best_metric['name'],
            'segment': segment,
            'metrics': {
                'mae': best_metric['mae'],
                'rmse': best_metric['mae'] * 1.5 # Placeholder for RMSE
            }
        })
        
    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_artifacts()
    app.run(debug=True, port=5000, host="0.0.0.0")
