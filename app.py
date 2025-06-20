from flask import Flask, request, jsonify, send_from_directory, render_template
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import traceback
from flask_cors import CORS
import os
import tensorflow as tf

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Load models and preprocessor
try:
    preprocessor = joblib.load('models/preprocessor.pkl')
    # Verify preprocessor is fitted
    if not hasattr(preprocessor, 'transform'):
        raise ValueError("Preprocessor is not fitted!")
    
    ml_models = {
        'Decision Tree': joblib.load('models/decision_tree.pkl'),
        'Logistic Regression': joblib.load('models/logistic_regression.pkl'),
        'Random Forest': joblib.load('models/random_forest.pkl')
    }
    cnn_model = tf.keras.models.load_model('models/cnn_fraud_detection.h5')
    print("All models loaded successfully")
except Exception as e:
    print(f"Error loading models: {str(e)}")
    print(traceback.format_exc())
    ml_models = {}
    cnn_model = None
    preprocessor = None

def preprocess_input(data):
    # Create DataFrame from input
    df = pd.DataFrame([data])
    
    # Feature engineering (must match training preprocessing)
    try:
        # Handle date parsing
        df['dob'] = pd.to_datetime(df['dob'], format='mixed')
        df['age'] = (datetime.now() - df['dob']).dt.days // 365
        
        # Calculate distance
        df['distance'] = np.sqrt((df['lat'] - df['merch_lat'])**2 + (df['long'] - df['merch_long'])**2)
        
        # Create features
        df['name_length'] = (df['first'] + df['last']).str.len()
        df['amount_per_age'] = df['amt'] / (df['age'] + 1)
        
        # Drop unnecessary columns
        cols_to_drop = ['first', 'last', 'street', 'city', 'state', 'zip', 'job', 'dob']
        df = df.drop([col for col in cols_to_drop if col in df.columns], axis=1)
        
        return df
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        print(traceback.format_exc())
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        print("Received data:", data)
        
        # Preprocess the input data
        processed_data = preprocess_input(data)
        
        # Transform using preprocessor
        X = preprocessor.transform(processed_data)
        
        # Convert sparse matrix to dense if needed
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        # Initialize results
        results = {}
        
        # Machine Learning predictions
        for model_name, model in ml_models.items():
            try:
                prediction = int(model.predict(X)[0])
                probability = float(model.predict_proba(X)[0][1])
                results[model_name] = {
                    'prediction': 'Fraud' if prediction == 1 else 'Not Fraud',
                    'probability': probability * 100
                }
            except Exception as e:
                print(f"Error in {model_name} prediction: {str(e)}")
                results[model_name] = {
                    'prediction': 'Error',
                    'probability': 0,
                    'error': str(e)
                }
        
        # CNN prediction
        if cnn_model:
            try:
                # Reshape for CNN
                X_cnn = X.reshape(X.shape[0], X.shape[1], 1)
                cnn_prediction = float(cnn_model.predict(X_cnn)[0][0])
                results['CNN'] = {
                    'prediction': 'Fraud' if cnn_prediction > 0.5 else 'Not Fraud',
                    'probability': cnn_prediction * 100
                }
            except Exception as e:
                print(f"Error in CNN prediction: {str(e)}")
                results['CNN'] = {
                    'prediction': 'Error',
                    'probability': 0,
                    'error': str(e)
                }
        
        return jsonify(results)
    
    except Exception as e:
        print(f"Error in prediction endpoint: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)