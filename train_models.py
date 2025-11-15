# train_models.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (classification_report, accuracy_score, precision_score, 
                           recall_score, f1_score, roc_auc_score, confusion_matrix, 
                           roc_curve, precision_recall_curve)
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import os
import json
import warnings
warnings.filterwarnings('ignore')

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('evaluation', exist_ok=True)

print("Loading dataset...")

try:
    # Try to load the dataset
    data = pd.read_csv('credit_card_transactions.csv')
    print("✅ Dataset loaded successfully")
    print(f"Dataset shape: {data.shape}")
    print(f"Columns: {data.columns.tolist()}")
    
    # Check dataset type and rename target variable if needed
    if 'Class' in data.columns:
        print("📊 Detected European Credit Card Dataset")
        data = data.rename(columns={'Class': 'is_fraud'})
        target_column = 'is_fraud'
    elif 'is_fraud' in data.columns:
        print("📊 Detected Custom Dataset")
        target_column = 'is_fraud'
    else:
        print("❓ Unknown dataset structure")
        # Try to guess target column
        if 'class' in data.columns:
            data = data.rename(columns={'class': 'is_fraud'})
            target_column = 'is_fraud'
        else:
            print("❌ No target column found. Please check your dataset.")
            exit()
    
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    exit()

# Display dataset info
print(f"\nTarget column: {target_column}")
print("Class distribution:")
print(data[target_column].value_counts())
print(f"Fraud rate: {data[target_column].mean():.6f}")

# Prepare features
# For European dataset, use V1-V28, Time, Amount
# For custom dataset, use all except target
if 'Time' in data.columns and 'Amount' in data.columns:
    # European dataset
    feature_columns = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount']
    print(f"Using European dataset features: {len(feature_columns)} features")
else:
    # Custom dataset - use all numeric columns except target
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    feature_columns = [col for col in numeric_columns if col != target_column]
    print(f"Using custom dataset features: {len(feature_columns)} numeric features")

print(f"Features: {feature_columns}")

# Separate features and target
X = data[feature_columns]
y = data[target_column]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, 'models/scaler.pkl')
print("✅ Scaler saved successfully")

def evaluate_model(model, model_name, X_test, y_test, y_pred, y_prob):
    """Comprehensive model evaluation"""
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # ROC curve data
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    
    # Precision-Recall curve data
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
    
    evaluation = {
        'model_name': model_name,
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'confusion_matrix': cm.tolist(),
        'roc_curve': {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist()
        },
        'pr_curve': {
            'precision': precision_curve.tolist(),
            'recall': recall_curve.tolist()
        }
    }
    
    return evaluation

def train_evaluate_model(model, model_name, X_train, y_train, X_test, y_test):
    print(f"\n🏃 Training {model_name}...")
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Comprehensive evaluation
    evaluation = evaluate_model(model, model_name, X_test, y_test, y_pred, y_prob)
    
    # Print results
    print(f"\n📊 {model_name} Results:")
    print("="*50)
    print(f"Accuracy:  {evaluation['accuracy']:.4f}")
    print(f"Precision: {evaluation['precision']:.4f}")
    print(f"Recall:    {evaluation['recall']:.4f}")
    print(f"F1 Score:  {evaluation['f1_score']:.4f}")
    print(f"ROC-AUC:   {evaluation['roc_auc']:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Save model
    model_file = f'models/{model_name.lower().replace(" ", "_")}.pkl'
    joblib.dump(model, model_file)
    print(f"💾 Model saved to {model_file}")
    
    return model, evaluation

# Calculate class weights
fraud_weight = len(y_train[y_train==0]) / len(y_train[y_train==1]) if len(y_train[y_train==1]) > 0 else 1
print(f"\n⚖️ Fraud class weight: {fraud_weight:.2f}")

# Store all evaluations
all_evaluations = {}

# 1. Random Forest
print("\n" + "="*50)
print("🌲 Training Random Forest Model")
print("="*50)
rf = RandomForestClassifier(
    n_estimators=100, 
    max_depth=10,
    random_state=42, 
    class_weight='balanced',
    n_jobs=-1
)
rf_model, rf_eval = train_evaluate_model(rf, 'Random Forest', X_train_scaled, y_train, X_test_scaled, y_test)
all_evaluations['Random Forest'] = rf_eval

# 2. XGBoost
print("\n" + "="*50)
print("🚀 Training XGBoost Model")
print("="*50)
xgb = XGBClassifier(
    scale_pos_weight=fraud_weight,
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100,
    random_state=42,
    eval_metric='logloss',
    n_jobs=-1
)
xgb_model, xgb_eval = train_evaluate_model(xgb, 'XGBoost', X_train_scaled, y_train, X_test_scaled, y_test)
all_evaluations['XGBoost'] = xgb_eval

# 3. Logistic Regression
print("\n" + "="*50)
print("📈 Training Logistic Regression Model")
print("="*50)
lr = LogisticRegression(
    max_iter=1000, 
    random_state=42, 
    class_weight='balanced'
)
lr_model, lr_eval = train_evaluate_model(lr, 'Logistic Regression', X_train_scaled, y_train, X_test_scaled, y_test)
all_evaluations['Logistic Regression'] = lr_eval

# 4. Neural Network
print("\n" + "="*50)
print("🧠 Training Neural Network Model")
print("="*50)

# Build model
nn_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# Compile model
nn_model.compile(
    optimizer=Adam(learning_rate=0.001), 
    loss='binary_crossentropy', 
    metrics=[
        'accuracy', 
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
    ]
)

# Train model
print("Training Neural Network model...")
history = nn_model.fit(
    X_train_scaled, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test_scaled, y_test),
    class_weight={0: 1, 1: fraud_weight},
    verbose=1
)

# Neural Network predictions and evaluation
y_pred_nn = (nn_model.predict(X_test_scaled, verbose=0) > 0.5).astype(int).flatten()
y_prob_nn = nn_model.predict(X_test_scaled, verbose=0).flatten()

nn_evaluation = evaluate_model(nn_model, 'Neural Network', X_test_scaled, y_test, y_pred_nn, y_prob_nn)

print("\n📊 Neural Network Results:")
print("="*50)
print(f"Accuracy:  {nn_evaluation['accuracy']:.4f}")
print(f"Precision: {nn_evaluation['precision']:.4f}")
print(f"Recall:    {nn_evaluation['recall']:.4f}")
print(f"F1 Score:  {nn_evaluation['f1_score']:.4f}")
print(f"ROC-AUC:   {nn_evaluation['roc_auc']:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_nn, zero_division=0))

all_evaluations['Neural Network'] = nn_evaluation

# Save Neural Network model
nn_model.save('models/nn_fraud_detection.h5')
print("💾 Neural network model saved successfully")

# Save all evaluations
with open('evaluation/model_evaluations.json', 'w') as f:
    json.dump(all_evaluations, f, indent=2)
print("💾 All evaluations saved to evaluation/model_evaluations.json")

# Create comparison table
print("\n" + "="*70)
print("🏆 MODEL COMPARISON SUMMARY")
print("="*70)

comparison_data = []
for model_name, eval_data in all_evaluations.items():
    comparison_data.append({
        'Model': model_name,
        'Accuracy': f"{eval_data['accuracy']:.4f}",
        'Precision': f"{eval_data['precision']:.4f}",
        'Recall': f"{eval_data['recall']:.4f}",
        'F1-Score': f"{eval_data['f1_score']:.4f}",
        'ROC-AUC': f"{eval_data['roc_auc']:.4f}"
    })

comparison_df = pd.DataFrame(comparison_data)
print("\n" + comparison_df.to_string(index=False))

# Find best model
best_f1 = max(all_evaluations.items(), key=lambda x: x[1]['f1_score'])
best_auc = max(all_evaluations.items(), key=lambda x: x[1]['roc_auc'])

print(f"\n🎯 BEST MODEL BY F1-Score: {best_f1[0]} (F1 = {best_f1[1]['f1_score']:.4f})")
print(f"🎯 BEST MODEL BY ROC-AUC: {best_auc[0]} (AUC = {best_auc[1]['roc_auc']:.4f})")

print("\n✅ All models trained and evaluated successfully!")