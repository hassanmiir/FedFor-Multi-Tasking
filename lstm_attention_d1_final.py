import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, TimeDistributed, Concatenate, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from sklearn.metrics import mean_absolute_error, mean_squared_error
import mlflow
import mlflow.keras

# Load the dataset
data = pd.read_csv("/home/disi/ff/Datasets/Dataset_1/forcasting.csv")
data['created_at'] = pd.to_datetime(data['created_at'])
data = data.sort_values(by='created_at')
forecast_cols = ['humidity (%)', 'temperature (DegCel)', 'light intensity']
df_forecast = data[forecast_cols]

# Scale the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_forecast)

def create_sequences(data, input_seq_length, output_seq_length):
    X, y = [], []
    for i in range(len(data) - input_seq_length - output_seq_length + 1):
        seq_x = data[i:(i + input_seq_length)]
        seq_y = data[(i + input_seq_length):(i + input_seq_length + output_seq_length)]
        X.append(seq_x)
        y.append(seq_y)  # Expecting each seq_y to be of length output_seq_length
    return np.array(X), np.array(y)

# Define sequence lengths
input_seq_length = 10
output_seq_length = 5  # Ensure this matches the sequence length in the target data
features = 3

# Create sequences
X, y = create_sequences(scaled_data, input_seq_length, output_seq_length)
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Clear previous Keras session
K.clear_session()

# Building the LSTM with Attention Model
input_layer = Input(shape=(input_seq_length, features))
x = LSTM(50, return_sequences=True)(input_layer)
x = Dropout(0.25)(x)

# Attention layer
attention = Attention()([x, x])  # Self-attention
x = Concatenate(axis=-1)([x, attention])

# Process combined output, adjust to output sequence length
x = LSTM(50, return_sequences=True)(x)
x = TimeDistributed(Dense(features, activation='linear'))(x)  # Output for each time step

model = Model(inputs=input_layer, outputs=x)
model.compile(optimizer='adam', loss='mse')
model.summary()

# Set the MLflow experiment
mlflow.set_experiment("LSTM_attention_final")

# Start MLflow run
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("input_seq_length", input_seq_length)
    mlflow.log_param("output_seq_length", output_seq_length)
    mlflow.log_param("epochs", 50)
    mlflow.log_param("batch_size", 64)

    # Training the Model
    history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test), verbose=1)

    # Log model
    mlflow.keras.log_model(model, "model")

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test.reshape(-1, y_test.shape[-1]), y_pred.reshape(-1, y_pred.shape[-1]))
    rmse = np.sqrt(mean_squared_error(y_test.reshape(-1, y_test.shape[-1]), y_pred.reshape(-1, y_pred.shape[-1])))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    # Log metrics
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("MAPE", mape)

    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
    print(f"MAPE: {mape}")
