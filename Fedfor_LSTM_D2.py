#Import Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Conv1D, MaxPooling1D, Dropout, Flatten, RepeatVector, Input, Attention
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

# File paths
train_file_1 = '/home/disi/ff/Datasets/Dataset_2/Train1.csv'
train_file_2 = '/home/disi/ff/Datasets/Dataset_2/Train2.csv'
test_file = '/home/disi/ff/Datasets/Dataset_2/Test.csv'

# Load the datasets
train_data_1 = pd.read_csv(train_file_1)
train_data_2 = pd.read_csv(train_file_2)
test_data = pd.read_csv(test_file)

# Combine training data and sort all data by 'created_at' if it exists
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

# Sequence parameters
input_seq_length = 10
output_seq_length = 3
features = 3  # Number of features

# Create sequences for training and testing
X_train, y_train = create_sequences(train_scaled, input_seq_length, output_seq_length)
X_test, y_test = create_sequences(test_scaled, input_seq_length, output_seq_length)

# Clear Keras session
K.clear_session()

# Model Architecture
input_layer = Input(shape=(input_seq_length, features))
x = Conv1D(64, (2), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.0001))(input_layer)
x = Dropout(0.2)(x)
x = Flatten()(x)
x = RepeatVector(output_seq_length)(x)
x = LSTM(50, activation='relu', dropout=0.2, return_sequences=True, kernel_regularizer=regularizers.l2(0.0001))(x)
output = TimeDistributed(Dense(3, activation='linear'))(x)
model = Model(inputs=input_layer, outputs=output)

# Optimizer
optimizer = tfa.optimizers.AdamW(learning_rate=0.0001, weight_decay=0.000001)

# Compile model
model.compile(optimizer=optimizer, loss='mse')

# Early stopping and learning rate reduction on plateau
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)

# Model summary
model.summary()

# MLflow Integration
mlflow.set_experiment("FedFor_LSTM_D2")
with mlflow.start_run():
    # Train the model
    history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test), verbose=1, callbacks=[early_stopping, reduce_lr])

    # Evaluate the model
    y_pred = model.predict(X_test)
    y_test_reshaped = y_test.reshape(-1, y_test.shape[-1])
    y_pred_reshaped = y_pred.reshape(-1, y_pred.shape[-1])

    # Calculate MAE, RMSE, and MAPE
    mae = mean_absolute_error(y_test_reshaped, y_pred_reshaped)
    rmse = np.sqrt(mean_squared_error(y_test_reshaped, y_pred_reshaped))
    mask = y_test_reshaped != 0
    mape = np.mean(np.abs((y_test_reshaped[mask] - y_pred_reshaped[mask]) / y_test_reshaped[mask])) * 100

    # Log metrics
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("MAPE", mape)

    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
    print(f"MAPE: {mape}%")
