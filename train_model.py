import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import resample
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, Dropout, Flatten, Dense

# Load dataset
df = pd.read_csv("fraudTrain.csv")

# Encode categorical columns
cat_cols = ['first', 'last', 'city', 'state', 'job', 'merchant', 'category', 'gender', 'street', 'trans_num']
encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Convert datetime
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], errors='coerce').astype(np.int64) // 10**9
df['dob'] = pd.to_datetime(df['dob'], errors='coerce').astype(np.int64) // 10**9
df.fillna(0, inplace=True)

# Balance dataset (undersampling)
df_majority = df[df.is_fraud == 0]
df_minority = df[df.is_fraud == 1]
df_majority_downsampled = resample(df_majority, replace=False, n_samples=len(df_minority)*3, random_state=42)
df_balanced = pd.concat([df_majority_downsampled, df_minority])

# Train/test split
X = df_balanced.drop('is_fraud', axis=1)
y = df_balanced['is_fraud']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# CNN model
model = Sequential([
    Conv1D(32, 2, activation='relu', input_shape=(X_scaled.shape[1], 1)),
    BatchNormalization(),
    Dropout(0.2),
    Conv1D(64, 2, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Flatten(),
    Dense(100, activation='relu'),
    Dropout(0.5),
    Dense(50, activation='relu'),
    Dense(25, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_test, y_test))

# Save everything
model.save("model/fraud_cnn.h5")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(encoders, "model/encoders.pkl")
joblib.dump(X.columns.tolist(), "model/feature_order.pkl")
