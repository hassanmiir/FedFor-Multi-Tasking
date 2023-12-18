#Import Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Conv1D, MaxPooling1D, Dropout, Flatten, RepeatVector, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
import tensorflow_addons as tfa
from sklearn.metrics import mean_absolute_error, mean_squared_error
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

def create_sequences(data, input_seq_length, output_seq_length):
    X, y = [], []
    for i in range(len(data) - input_seq_length - output_seq_length + 1):
        seq_x = data[i:i + input_seq_length]
        seq_y = data[i + input_seq_length:i + input_seq_length + output_seq_length]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

input_seq_length = 10
features = 3
output_seq_length = 3
X, y = create_sequences(scaled_data, input_seq_length, output_seq_length)
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

K.clear_session()

input_layer = Input(shape=(input_seq_length, features))

# Convolutional layers with BatchNormalization and regularization
x = Conv1D(16, (2), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001))(input_layer)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=(2))(x)
x = Conv1D(32, (2), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
x = BatchNormalization()(x)
x = Conv1D(64, (2), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
x = Dropout(0.2)(x)
x = Flatten()(x)
x = RepeatVector(output_seq_length)(x)

# LSTM layers with dropout
x = LSTM(100, activation='relu', dropout=0.2, return_sequences=True, kernel_regularizer=regularizers.l2(0.001))(x)
output = TimeDistributed(Dense(3, activation = 'sigmoid'))(x)

model = Model(inputs=input_layer, outputs=output)

# Optimizer with custom learning rate
#optimizer = Adam(learning_rate=0.0001, weight_decay=1e-5)
optimizer = tfa.optimizers.AdamW(learning_rate=0.0001, weight_decay=0.000001)

# Compile model with regularization loss
model.compile(optimizer=optimizer, loss='mse')

# Early stopping and learning rate reduction on plateau
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)

model.summary()

# Set the MLflow experiment
mlflow.set_experiment("FedFor_LSTM_D1")

    #Implement early stopping
    #early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    # Start an MLflow run
with mlflow.start_run():
    # Train the model with early stopping
    history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test), verbose=1, callbacks=[early_stopping, reduce_lr])

   # Evaluate the model
    y_pred = model.predict(X_test)

    # Reshape the data for evaluation
    y_test_reshaped = y_test.reshape(-1, y_test.shape[-1])
    y_pred_reshaped = y_pred.reshape(-1, y_pred.shape[-1])

    # Calculate MAE, RMSE, and MAPE
    mae = mean_absolute_error(y_test_reshaped, y_pred_reshaped)
    rmse = np.sqrt(mean_squared_error(y_test_reshaped, y_pred_reshaped))

    # For MAPE, ensure we don't divide by zero
    mask = y_test_reshaped != 0
    mape = np.mean(np.abs((y_test_reshaped[mask] - y_pred_reshaped[mask]) / y_test_reshaped[mask])) * 100

    # Log metrics
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("MAPE", mape)

    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
    print(f"MAPE: {mape}%")