import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

from flask import Flask, request, jsonify, render_template
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
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    preprocessor = joblib.load(os.path.join(models_dir, 'preprocessor.pkl'))
    
    if not hasattr(preprocessor, 'transform'):
        raise ValueError("Preprocessor is not fitted!")
    
    ml_models = {
        'Decision Tree': joblib.load(os.path.join(models_dir, 'decision_tree.pkl')),
        'Logistic Regression': joblib.load(os.path.join(models_dir, 'logistic_regression.pkl')),
        'Random Forest': joblib.load(os.path.join(models_dir, 'random_forest.pkl'))
    }
    cnn_model = tf.keras.models.load_model(os.path.join(models_dir, 'cnn_fraud_detection.h5'))
    print("All models loaded successfully")
except Exception as e:
    print(f"Error loading models: {str(e)}")
    print(traceback.format_exc())
    ml_models = {}
    cnn_model = None
    preprocessor = None

def preprocess_input(data):
    # Create DataFrame from input with proper type conversion
    df = pd.DataFrame([{
        'cc_num': str(data.get('cc_num', '')),
        'merchant': str(data.get('merchant', '')),
        'category': str(data.get('category', '')),
        'amt': float(data.get('amt', 0)),
        'first': str(data.get('first', '')),
        'last': str(data.get('last', '')),
        'gender': str(data.get('gender', 'M')),
        'street': str(data.get('street', '')),
        'city': str(data.get('city', '')),
        'state': str(data.get('state', '')),
        'zip': str(data.get('zip', '')),
        'lat': float(data.get('lat', 0)),
        'long': float(data.get('long', 0)),
        'job': str(data.get('job', '')),
        'dob': data.get('dob', ''),
        'merch_lat': float(data.get('merch_lat', 0)),
        'merch_long': float(data.get('merch_long', 0)),
        'trans_date_trans_time': data.get('trans_date_trans_time', ''),
        'is_fraud': 0
    }])
    
    try:
        # Handle date parsing
        df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
        df['age'] = (datetime.now() - df['dob']).dt.days // 365
        
        # Calculate distance
        df['distance'] = np.sqrt(
            (df['lat'] - df['merch_lat'])**2 + 
            (df['long'] - df['merch_long'])**2
        )
        
        # Create features with explicit type handling
        df['name_length'] = (df['first'].astype(str) + df['last'].astype(str)).str.len()
        df['amount_per_age'] = df['amt'] / (df['age'].replace(0, 1))  # Avoid division by zero
        
        # Drop unnecessary columns
        cols_to_drop = ['first', 'last', 'street', 'city', 'state', 'zip', 'job', 'dob']
        df = df.drop([col for col in cols_to_drop if col in df.columns], axis=1)
        
        return df
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        print(traceback.format_exc())
        raise ValueError(f"Preprocessing failed: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/metrics')
def show_metrics():
    try:
        with open('evaluation/model_evaluations.json', 'r') as f:
            evaluations = json.load(f)
        
        # Create a simplified version for the web
        web_metrics = {}
        for model_name, metrics in evaluations.items():
            web_metrics[model_name] = {
                'accuracy': round(metrics['accuracy'] * 100, 2),
                'precision': round(metrics['precision'] * 100, 2),
                'recall': round(metrics['recall'] * 100, 2),
                'f1_score': round(metrics['f1_score'] * 100, 2),
                'roc_auc': round(metrics['roc_auc'] * 100, 2)
            }
        
        return jsonify(web_metrics)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
        
        return jsonify(results), 200
    
    except Exception as e:
        print(f"Error in prediction endpoint: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'trace': traceback.format_exc()
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

