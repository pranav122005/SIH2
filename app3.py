# ===========================================================
# SATELLITE ERROR FORECASTING USING LSTM
# ===========================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# --------- 1. Load dataset ---------
file_path = "meo2_training.txt"  # change to geo_training.txt or meo_training.txt as needed
data = pd.read_csv(file_path, sep="\t")

# Inspect columns
print("Columns:", data.columns)

# --------- 2. Preprocess ---------
# Convert utc_time to datetime
data['utc_time'] = pd.to_datetime(data['utc_time'], errors='coerce')
data = data.dropna(subset=['utc_time'])

# Sort by time (important for time series)
data = data.sort_values('utc_time')

# Set index to utc_time
data.set_index('utc_time', inplace=True)

# Select one column to predict, e.g., satellite clock error
target_col = 'satclockerror (m)' if 'satclockerror (m)' in data.columns else 'satclockerror'

# Fill missing values by interpolation
data = data.interpolate()

# Normalize all colu
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[[target_col]])

# --------- 3. Create sequences ---------
def create_sequences(series, n_steps=10):
    X, y = [], []
    for i in range(len(series) - n_steps):
        X.append(series[i:i+n_steps])
        y.append(series[i+n_steps])
    return np.array(X), np.array(y)

n_steps = 10  # use last 10 time steps to predict next
X, y = create_sequences(scaled_data, n_steps)
print("X shape:", X.shape, "y shape:", y.shape)

# --------- 4. Train-test split ---------
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# --------- 5. Build LSTM model ---------
model = Sequential([
    LSTM(64, activation='tanh', input_shape=(n_steps, 1)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# --------- 6. Train model ---------
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=16,
    callbacks=[early_stop],
    verbose=1
)

# --------- 7. Make predictions ---------
pred_scaled = model.predict(X_test)
# Inverse transform to original scale
y_pred = scaler.inverse_transform(pred_scaled)
y_true = scaler.inverse_transform(y_test)

# --------- 8. Plot results ---------
plt.figure(figsize=(10,5))
plt.plot(y_true, label='True Error', color='blue')
plt.plot(y_pred, label='Predicted Error', color='red')
plt.title(f"LSTM Prediction for {target_col}")
plt.xlabel("Time steps")
plt.ylabel("Error (m)")
plt.legend()
plt.show()

# --------- 9. Evaluate ---------
from sklearn.metrics import mean_squared_error, mean_absolute_error
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
print(f"RMSE: {rmse:.6f}, MAE: {mae:.6f}")
