import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Enable GPU acceleration if available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Load dataset
try:
    data = pd.read_csv('fraudTrain.csv')
    print("Dataset loaded successfully")
    print(f"Original shape: {data.shape}")
    
    # Clean column names
    data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
    print("\nCleaned column names:", data.columns.tolist())
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Feature engineering
def preprocess_data(df):
    # Convert date of birth to age
    try:
        df['dob'] = pd.to_datetime(df['dob'], format='mixed')
        df['age'] = (datetime.now() - df['dob']).dt.days // 365
    except Exception as e:
        print(f"Error converting dob: {e}")
        raise
    
    # Calculate distance between customer and merchant
    df['distance'] = np.sqrt((df['lat'] - df['merch_lat'])**2 + (df['long'] - df['merch_long'])**2)
    
    # Extract useful features
    df['name_length'] = (df['first'] + df['last']).str.len()
    df['amount_per_age'] = df['amt'] / (df['age'] + 1)  # Add 1 to avoid division by zero
    
    # Drop original columns we've transformed
    cols_to_drop = ['first', 'last', 'street', 'city', 'state', 'zip', 'job', 'dob']
    df = df.drop([col for col in cols_to_drop if col in df.columns], axis=1)
    
    return df

# Preprocess the data
try:
    processed_data = preprocess_data(data)
    print("\nData preprocessing completed successfully")
    print(f"Processed shape: {processed_data.shape}")
except Exception as e:
    print(f"Error during preprocessing: {e}")
    exit()

# Separate features and target
X = processed_data.drop('is_fraud', axis=1)
y = processed_data['is_fraud']

# Print class distribution
print("\nClass distribution:")
print(y.value_counts())
print(f"Fraud rate: {y.mean():.4f}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define preprocessing steps
numeric_features = ['amt', 'lat', 'long', 'merch_lat', 'merch_long', 'age', 'distance', 'name_length', 'amount_per_age']
categorical_features = ['category', 'gender']
high_cardinality_features = ['merchant', 'cc_num']

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

high_cardinality_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
        ('high_card', high_cardinality_transformer, high_cardinality_features)
    ])

# Fit and save preprocessor
print("\nFitting preprocessor...")
preprocessor.fit(X_train)
joblib.dump(preprocessor, 'models/preprocessor.pkl')
print("Preprocessor fitted and saved successfully")

# Transform data
print("\nTransforming data...")
X_train_transformed = preprocessor.transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

print(f"Training shape: {X_train_transformed.shape}")
print(f"Test shape: {X_test_transformed.shape}")

def train_evaluate_model(model, model_name, X_train, y_train, X_test, y_test):
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)
    
    # Evaluation
    print(f"\n{model_name} Results:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")
    
    # Save model
    model_file = f'models/{model_name.lower().replace(" ", "_")}.pkl'
    joblib.dump(model, model_file)
    print(f"Model saved to {model_file}")
    
    return model

# Calculate class weights for imbalanced data
fraud_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
print(f"\nFraud class weight: {fraud_weight:.2f}")

# 1. Decision Tree
print("\n" + "="*50)
print("Training Decision Tree Model")
print("="*50)
dt = DecisionTreeClassifier(max_depth=8,
                          min_samples_split=10,
                          class_weight='balanced',
                          random_state=42)
train_evaluate_model(dt, 'Decision Tree', X_train_transformed, y_train, X_test_transformed, y_test)

# 2. Logistic Regression
print("\n" + "="*50)
print("Training Logistic Regression Model")
print("="*50)
lr = LogisticRegression(class_weight='balanced',
                       random_state=42,
                       max_iter=1000,
                       solver='liblinear',
                       penalty='l2',
                       n_jobs=1)
train_evaluate_model(lr, 'Logistic Regression', X_train_transformed, y_train, X_test_transformed, y_test)

# 3. Random Forest
print("\n" + "="*50)
print("Training Random Forest Model")
print("="*50)
rf = RandomForestClassifier(n_estimators=200, 
                           max_depth=10,
                           random_state=42, 
                           class_weight='balanced',
                           n_jobs=-1)
train_evaluate_model(rf, 'Random Forest', X_train_transformed, y_train, X_test_transformed, y_test)

# 4. CNN (Optimized Version)
print("\n" + "="*50)
print("Training CNN Model")
print("="*50)

# Convert to dense arrays
X_train_transformed = X_train_transformed.toarray() if hasattr(X_train_transformed, 'toarray') else X_train_transformed
X_test_transformed = X_test_transformed.toarray() if hasattr(X_test_transformed, 'toarray') else X_test_transformed

# Reshape for CNN
X_train_cnn = X_train_transformed.reshape(X_train_transformed.shape[0], X_train_transformed.shape[1], 1)
X_test_cnn = X_test_transformed.reshape(X_test_transformed.shape[0], X_test_transformed.shape[1], 1)

# Simplified CNN Model
cnn_model = Sequential([
    Conv1D(32, 3, activation='relu', input_shape=(X_train_cnn.shape[1], 1)),
    MaxPooling1D(2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# Optimized Compile
cnn_model.compile(optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

# Callbacks for better training
callbacks = [
    EarlyStopping(patience=2, monitor='val_auc', mode='max', restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2)
]

print("Training CNN model...")
history = cnn_model.fit(
    X_train_cnn, y_train,
    epochs=3,
    batch_size=1024,
    validation_data=(X_test_cnn, y_test),
    class_weight={0: 1, 1: fraud_weight},
    callbacks=callbacks,
    verbose=1
)

# Evaluate model
print("\nCNN Results:")
eval_results = cnn_model.evaluate(X_test_cnn, y_test)
print(f"Accuracy: {eval_results[1]:.4f}")
print(f"AUC: {eval_results[2]:.4f}")

# Save model
cnn_model.save('models/cnn_fraud_detection.h5')
print("\nCNN model saved successfully")

print("\nAll models trained and saved successfully!")