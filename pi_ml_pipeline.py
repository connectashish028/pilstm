import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# ========== CONFIGURATION ==========
# Set up directories
ASSETS_DIR = 'assets'
os.makedirs(ASSETS_DIR, exist_ok=True)

# Set this to True if you want to use InfluxDB, otherwise dummy data will be used
use_influxdb = True

# InfluxDB credentials (replace with your own, do NOT commit sensitive info)
INFLUXDB_HOST = 'YOUR_HOST'  # e.g., '192.168.1.100'
INFLUXDB_PORT = 8086
INFLUXDB_USER = 'YOUR_USERNAME'
INFLUXDB_PASS = 'YOUR_PASSWORD'
INFLUXDB_DB = 'YOUR_DB_NAME'

# ========== DATA INGESTION ==========
if use_influxdb:
    from influxdb import InfluxDBClient
    client = InfluxDBClient(
        host=INFLUXDB_HOST,
        port=INFLUXDB_PORT,
        username=INFLUXDB_USER,
        password=INFLUXDB_PASS,
        database=INFLUXDB_DB
    )
    query = 'SELECT "temperature_C", "humidity" FROM "environment"'
    result = client.query(query)
    data = list(result.get_points(measurement='environment'))
    df = pd.DataFrame(data)
    # Convert time and set datetime index
    df['time'] = df['time'].str.replace('Z', '+00:00')
    df['time'] = pd.to_datetime(df['time'], format="mixed")
else:
    # For demonstration, create dummy data
    dates = pd.date_range('2023-01-01', periods=2000, freq='min')
    temp = 20 + 5 * np.sin(np.linspace(0, 20, 2000)) + np.random.normal(0, 0.5, 2000)
    hum = 50 + 10 * np.cos(np.linspace(0, 10, 2000)) + np.random.normal(0, 1, 2000)
    df = pd.DataFrame({'time': dates, 'temperature_C': temp, 'humidity': hum})
    df['time'] = pd.to_datetime(df['time'])
# Ensure datetime index for resampling
df.set_index('time', inplace=True)
df.sort_index(inplace=True)

# Resample to 1-minute frequency (mean aggregation)
df_1min = df.resample('1min').mean()

# Interpolate missing values
df_1min = df_1min.interpolate(method='linear')

# Drop outliers (DHT22 limits)
df_1min = df_1min[(df_1min['temperature_C'].between(-40, 80)) & 
                  (df_1min['humidity'].between(0, 100))]

# ========== EDA ==========
fig, ax1 = plt.subplots(figsize=(12, 5))
color_temp = 'tab:red'
ax1.set_xlabel("Time")
ax1.set_ylabel("Temperature (°C)", color=color_temp)
temp_line, = ax1.plot(df_1min.index, df_1min['temperature_C'], color=color_temp, label="Temperature (°C)")
ax1.tick_params(axis='y', labelcolor=color_temp)
ax1.grid(True)

ax2 = ax1.twinx()
color_hum = 'tab:blue'
ax2.set_ylabel("Humidity (%)", color=color_hum)
hum_line, = ax2.plot(df_1min.index, df_1min['humidity'], color=color_hum, label="Humidity (%)")
ax2.tick_params(axis='y', labelcolor=color_hum)

lines = [temp_line, hum_line]
labels = [str(line.get_label()) for line in lines]
ax1.legend(lines, labels, loc='upper left')
plt.title("Temperature and Humidity Over Time")
plt.tight_layout()
eda_plot_path = os.path.join(ASSETS_DIR, 'eda_plot.png')
plt.savefig(eda_plot_path)
plt.close()

# ========== MODELING ==========
df = df_1min.copy()
df = df.sort_index()
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df.values)

def create_sequences(data, window_size=60):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

window_size = 60
X, y = create_sequences(scaled, window_size)
n = X.shape[0]
t1 = int(n*0.70); t2 = int(n*0.85)
X_train, y_train = X[:t1], y[:t1]
X_val,   y_val   = X[t1:t2], y[t1:t2]
X_test,  y_test  = X[t2:],   y[t2:]

import tensorflow as tf

# Ensure TensorFlow uses the GPU if available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU(s): {[gpu.name for gpu in physical_devices]}")
    except Exception as e:
        print(f"Could not set GPU memory growth: {e}")
else:
    print("No GPU found, using CPU.")

model = Sequential([
    LSTM(50, input_shape=(window_size, X.shape[2])),
    Dense(X.shape[2])
])
model.compile(optimizer='adam', loss='mse')
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train,
          validation_data=(X_val, y_val),
          epochs=100, batch_size=32, callbacks=[es], verbose=2)

y_pred_scaled = model.predict(X_test)
y_test_inv = scaler.inverse_transform(y_test)
y_pred_inv = scaler.inverse_transform(y_pred_scaled)
time_index = df.index[window_size + t2 : window_size + t2 + len(y_test_inv)]

# ========== PLOTS: Actual vs Predicted ==========
plt.figure(figsize=(12,5))
plt.subplot(2,1,1)
plt.plot(time_index, y_test_inv[:,0], label='Actual Temp (°C)')
plt.plot(time_index, y_pred_inv[:,0], linestyle='--', label='Predicted Temp (°C)')
plt.title('Temperature: Actual vs Predicted')
plt.legend()
plt.ylabel('°C')
plt.subplot(2,1,2)
plt.plot(time_index, y_test_inv[:,1], label='Actual Humidity (%)')
plt.plot(time_index, y_pred_inv[:,1], linestyle='--', label='Predicted Humidity (%)')
plt.title('Humidity: Actual vs Predicted')
plt.legend()
plt.ylabel('%RH')
plt.xlabel('Time')
plt.tight_layout()
temp_pred_path = os.path.join(ASSETS_DIR, 'temp_pred.png')
hum_pred_path = os.path.join(ASSETS_DIR, 'hum_pred.png')
plt.savefig(temp_pred_path)
plt.close()

# ========== PLOT: Training vs Validation Loss ==========
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
loss_plot_path = os.path.join(ASSETS_DIR, 'loss_plot.png')
plt.savefig(loss_plot_path)
plt.close()

# ========== METRICS ==========
mse_temp = mean_squared_error(y_test_inv[:, 0], y_pred_inv[:, 0])
mae_temp = mean_absolute_error(y_test_inv[:, 0], y_pred_inv[:, 0])
rmse_temp = np.sqrt(mse_temp)
mse_hum = mean_squared_error(y_test_inv[:, 1], y_pred_inv[:, 1])
mae_hum = mean_absolute_error(y_test_inv[:, 1], y_pred_inv[:, 1])
rmse_hum = np.sqrt(mse_hum)

variables = ['Temperature', 'Humidity']
metrics = {
    'MSE': [mse_temp, mse_hum],
    'MAE': [mae_temp, mae_hum],
    'RMSE': [rmse_temp, rmse_hum]
}
x = np.arange(len(variables))
width = 0.25
plt.figure()
for i, (metric_name, values) in enumerate(metrics.items()):
    plt.bar(x + i*width, values, width, label=metric_name)
plt.title('Performance Metrics on Test Set')
plt.xlabel('Variable')
plt.ylabel('Value')
plt.xticks(x + width, variables)
plt.legend()
metrics_bar_path = os.path.join(ASSETS_DIR, 'metrics_bar.png')
plt.savefig(metrics_bar_path)
plt.close()
