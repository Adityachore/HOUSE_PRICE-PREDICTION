from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Paths
MODEL_PATH = 'house_price_model.pkl'
COLUMNS_PATH = 'model_columns.pkl'

def load_resources():
    if os.path.exists(MODEL_PATH) and os.path.exists(COLUMNS_PATH):
        return joblib.load(MODEL_PATH), joblib.load(COLUMNS_PATH)
    return None, None

model, model_columns = load_resources()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global model, model_columns
    if model is None or model_columns is None:
        model, model_columns = load_resources()
        if model is None:
            return jsonify({'error': 'Model not found. Please train the model first.'}), 500
            
    try:
        data = request.get_json()
        
        # 1. Base Numeric Features
        # Mapping UI checkboxes to model inputs
        # Garage: Check -> 1 car (standard), Uncheck -> 0
        # Pool: Check -> 500 sq ft (standard pool), Uncheck -> 0
        input_data = {
            'GrLivArea': data.get('area', 1500),
            'BedroomAbvGr': data.get('bedrooms', 3),
            'FullBath': data.get('bathrooms', 2),
            'YearBuilt': data.get('yearBuilt', 2000),
            'LotArea': data.get('lotArea', 5000),
            'ExterQual_Score': data.get('condition_score', 2),
            'GarageCars': 1 if data.get('hasGarage') else 0,
            'PoolArea': 500 if data.get('hasPool') else 0
        }
        
        # 2. Handle One-Hot Encoding for Zoning (Location Type)
        selected_zone = data.get('location', 'RL')
        
        # Initialize all model columns with 0
        df_input = pd.DataFrame(columns=model_columns)
        df_input.loc[0] = 0
        
        # Fill numeric values
        for col in input_data:
            if col in model_columns:
                df_input.loc[0, col] = input_data[col]
        
        # Set the one-hot column to 1
        zone_col = f"Zone_{selected_zone}"
        if zone_col in model_columns:
            df_input.loc[0, zone_col] = 1
        
        # Make prediction
        predicted_price = model.predict(df_input)[0]
        
        # Ensure price is non-negative
        predicted_price = max(0, predicted_price)
        
        return jsonify({'predicted_price': float(predicted_price)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
