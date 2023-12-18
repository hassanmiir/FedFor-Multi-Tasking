# Import Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GRU, Dense, Input
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
import mlflow
import mlflow.keras

# Load the dataset
data = pd.read_csv("/home/disi/ff/Datasets/Dataset_1/forcasting.csv")
data['created_at'] = pd.to_datetime(data['created_at'])
data = data.sort_values(by='created_at')

forecast_cols = ['humidity (%)', 'temperature (DegCel)', 'light intensity']
df_forecast = data[forecast_cols]
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_forecast)

# Function to create sequences
def create_sequences(data, input_seq_length, output_seq_length):
    X, y = [], []
    for i in range(len(data) - input_seq_length - output_seq_length + 1):
        seq_x = data[i:i + input_seq_length]
        seq_y = data[i + input_seq_length:i + input_seq_length + output_seq_length]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Sequence parameters
input_seq_length = 10
output_seq_length = 5
features = 3  # Number of features in the input/output data

# Create sequences
X, y = create_sequences(scaled_data, input_seq_length, output_seq_length)

# Split the data into training and testing sets
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Clear Keras session
tf.keras.backend.clear_session()

# Model architecture
input_layer = Input(shape=(input_seq_length, features))
gru = GRU(50, return_sequences=False)(input_layer)
output = Dense(output_seq_length * features, activation='linear')(gru)
output = tf.keras.layers.Reshape((output_seq_length, features))(output)

model = Model(inputs=input_layer, outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Model summary
model.summary()

# Start MLflow run
mlflow.set_experiment("GRU_Time_Series_Forecasting")
with mlflow.start_run():
    # Train the model
    history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test), verbose=1)

    # Evaluate the model
    y_pred = model.predict(X_test)

    # Reshape for evaluation
    y_test_reshaped = y_test.reshape(-1, y_test.shape[-1])
    y_pred_reshaped = y_pred.reshape(-1, y_pred.shape[-1])

    # Calculate MAE, RMSE, and MAPE
    mae = mean_absolute_error(y_test_reshaped, y_pred_reshaped)
    rmse = np.sqrt(mean_squared_error(y_test_reshaped, y_pred_reshaped))
    mape = np.mean(np.abs((y_test_reshaped - y_pred_reshaped) / y_test_reshaped)) * 100

    # Log metrics to MLflow
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("MAPE", mape)

    # Log model
    mlflow.keras.log_model(model, "model")

    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
    print(f"MAPE: {mape}%")
