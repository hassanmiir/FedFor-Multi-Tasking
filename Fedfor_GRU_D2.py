# Import Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, GRU, Dense, Conv1D, MaxPooling1D, Dropout, Flatten, RepeatVector, TimeDistributed, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow_addons as tfa
from tensorflow_addons.optimizers import AdamW
import mlflow
import mlflow.keras

# File paths
train_file_1 = '/home/disi/ff/Datasets/Dataset_2/Train1.csv'
train_file_2 = '/home/disi/ff/Datasets/Dataset_2/Train2.csv'
test_file = '/home/disi/ff/Datasets/Dataset_2/Test.csv'


# Load the datasets
train_data_1 = pd.read_csv(train_file_1)
train_data_2 = pd.read_csv(train_file_2)
test_data = pd.read_csv(test_file)

# Combine training data and sort all data by 'Date' if it exists
train_data = pd.concat([train_data_1, train_data_2], ignore_index=True)
if 'Date' in train_data.columns:
    train_data['Date'] = pd.to_datetime(train_data['Date'])
    train_data = train_data.sort_values(by='Date')
if 'Date' in test_data.columns:
    test_data['Date'] = pd.to_datetime(test_data['Date'])
    test_data = test_data.sort_values(by='Date')

# Selecting the relevant columns
forecast_cols = ['Temperature', 'Humidity', 'Light']
train_forecast = train_data[forecast_cols]
test_forecast = test_data[forecast_cols]

# Scale the data
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_forecast)
test_scaled = scaler.transform(test_forecast)

# Function to create sequences
def create_sequences(data, input_seq_length, output_seq_length):
    X, y = [], []
    for i in range(len(data) - input_seq_length - output_seq_length + 1):
        seq_x = data[i:i + input_seq_length]
        seq_y = data[i + input_seq_length:i + input_seq_length + output_seq_length]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Sequence lengths and features
input_seq_length = 10
output_seq_length = 3
features = train_forecast.shape[1]

# Create sequences
X, y = create_sequences(train_scaled, input_seq_length, output_seq_length)
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Clear previous Keras session
from tensorflow.keras import backend as K
K.clear_session()

# Model architecture
input_layer = Input(shape=(input_seq_length, features))
x = Conv1D(16, 2, activation='relu', padding='same')(input_layer)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2)(x)
x = Dropout(0.25)(x)
x = Flatten()(x)
x = RepeatVector(output_seq_length)(x)
x = GRU(50, activation='relu', dropout=0.25, return_sequences=True)(x)
output = TimeDistributed(Dense(features, activation='linear'))(x)
model = Model(inputs=input_layer, outputs=output)

# Compile the model using AdamW optimizer with weight decay
optimizer = AdamW(learning_rate=0.0001, weight_decay=0.000002)
model.compile(optimizer=optimizer, loss='mse')
model.summary()

# Set the MLflow experiment
mlflow.set_experiment("FedFor_GRU_D2")

# Train the model
with mlflow.start_run():
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

    # Evaluate the model
    y_pred = model.predict(X_test)

    # Reshape data for evaluation
    y_test_reshaped = y_test.reshape(-1, y_test.shape[-1])
    y_pred_reshaped = y_pred.reshape(-1, y_pred.shape[-1])

    # Calculate metrics
    mae = mean_absolute_error(y_test_reshaped, y_pred_reshaped)
    rmse = np.sqrt(mean_squared_error(y_test_reshaped, y_pred_reshaped))
    mask = y_test_reshaped != 0
    mape = np.mean(np.abs((y_test_reshaped[mask] - y_pred_reshaped[mask]) / y_test_reshaped[mask])) * 100

    # Log metrics in MLflow
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("MAPE", mape)

    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
    print(f"MAPE: {mape}%")
