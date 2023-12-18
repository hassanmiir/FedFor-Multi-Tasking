import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras_self_attention import SeqSelfAttention
import mlflow
import mlflow.keras
from math import sqrt
import joblib

# MLflow tracking URI set to local host
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("LSTM_with_Attention_Forecasting")

# Function to calculate MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Load the dataset
data = pd.read_csv('/home/disi/ff/Datasets/Dataset_1/forcasting.csv')

# Select relevant features
features = data[['humidity (%)', 'temperature (DegCel)', 'light intensity']]

# Normalize features
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

# Save the scaler to disk
joblib.dump(scaler, 'scaler.pkl')

# Define a function to create samples from time series data
def create_dataset(X, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(X.iloc[i + time_steps].values)
    return np.array(Xs), np.array(ys)

# Create sequences of 5 timesteps
time_steps = 15
X, y = create_dataset(pd.DataFrame(scaled_features), time_steps)

# Split the data into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


# 1. Building the LSTM with Attention Model
sequence_length = 10
num_features = 3

input_layer = Input(shape=(sequence_length, num_features))

# LSTM layers
x = LSTM(50, return_sequences=True)(input_layer)
x = Dropout(0.25)(x)

# Attention layer
x = Attention()([x, x])

x = Dense(50, activation='relu')(x)
output = Dense(3, activation='sigmoid')(x)

model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer='adam', loss='mse')

model.summary()

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Start MLflow run
with mlflow.start_run(nested=True):
    # Train the model
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=1)

    # Predict and inverse transform to original scale
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_true = scaler.inverse_transform(y_test)

    # Calculate the losses
    mae = mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)

    # Log metrics
    mlflow.log_metric('MAE', mae)
    mlflow.log_metric('RMSE', rmse)
    mlflow.log_metric('MAPE', mape)

    # Log model
    mlflow.keras.log_model(model, "LSTM_Attention_Model")

    # Log scaler
    mlflow.log_artifact('scaler.pkl')

# Print the losses
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Root Mean Square Error (RMSE): {rmse}')
print(f'Mean Absolute Percentage Error (MAPE): {mape}')
