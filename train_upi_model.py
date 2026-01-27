# =============================================================
# UPI TRANSACTION ANOMALY DETECTION - TRAIN & SAVE OBJECTS
# =============================================================

# -----------------------------
# Step 0: Install Libraries
# -----------------------------
# Only if not already installed
# pip install pandas numpy scikit-learn tensorflow joblib

# -----------------------------
# Step 1: Import Libraries
# -----------------------------
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import joblib

# -----------------------------
# Step 2: Load Dataset
# -----------------------------
file_path = "upi_transactions_reduced.csv"  # <-- replace with your dataset path
df = pd.read_csv(file_path)
print("✅ Dataset loaded successfully. Sample:")
print(df.head())

# -----------------------------
# Step 3: Feature Engineering
# -----------------------------
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour_of_day'] = df['timestamp'].dt.hour
df['log_amount'] = np.log1p(df['amount'])

categorical_cols = ['transaction_type', 'location', 'device_type', 'transaction_status']
numerical_cols = ['log_amount', 'hour_of_day']

# -----------------------------
# Step 4: Preprocessing
# -----------------------------
ct = ColumnTransformer(
    [('ohe', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)],
    remainder='passthrough'
)
X = ct.fit_transform(df[categorical_cols + numerical_cols])
y = df['label'].values

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# Step 5: Save Preprocessing Objects
# -----------------------------
joblib.dump(ct, "ct.pkl")
joblib.dump(scaler, "scaler.pkl")
print("✅ ct.pkl and scaler.pkl saved successfully!")

# -----------------------------
# Step 6: Split Training Data
# -----------------------------
X_train = X_scaled[y == 0]  # train only on normal transactions

# -----------------------------
# Step 7: Build Autoencoder
# -----------------------------
input_dim = X_train.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(32, activation='relu')(input_layer)
encoded = Dense(16, activation='relu')(encoded)
encoded = Dense(8, activation='relu')(encoded)
decoded = Dense(16, activation='relu')(encoded)
decoded = Dense(32, activation='relu')(decoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
autoencoder.summary()

# -----------------------------
# Step 8: Train Autoencoder
# -----------------------------
history = autoencoder.fit(
    X_train, X_train,
    epochs=50,        # increase for better accuracy
    batch_size=32,
    validation_split=0.2,
    shuffle=True
)

# -----------------------------
# Step 9: Compute Reconstruction Errors & Threshold
# -----------------------------
X_train_pred = autoencoder.predict(X_train)
mse = np.mean(np.power(X_train - X_train_pred, 2), axis=1)
threshold = np.percentile(mse, 95)  # 95th percentile
joblib.dump(threshold, "threshold.pkl")
print(f"✅ threshold.pkl saved. Threshold value: {threshold:.6f}")

# -----------------------------
# Step 10: Save Autoencoder Model
# -----------------------------
autoencoder.save("upi_autoencoder_model.keras")
print("✅ Autoencoder model saved as upi_autoencoder_model.keras")

print("\n🎉 All required files are ready for Streamlit app!")
