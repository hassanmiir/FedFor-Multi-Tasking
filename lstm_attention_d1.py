#Import Libraries
import pandas as pd
import numpy as np
from attention import Attention
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, LSTM, Dense, TimeDistributed, Conv1D, MaxPooling1D, Dropout, Flatten, RepeatVector
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
import tensorflow_addons as tfa
from imblearn.over_sampling import SMOTE
from keras.layers import Attention

import mlflow
import mlflow.keras

# MLflow experiment
mlflow.start_run()

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
output_seq_length = 5
X, y = create_sequences(scaled_data, input_seq_length, output_seq_length)
train_size = int(0.7 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

K.clear_session()

input_layer = Input(shape=(input_seq_length, features))

# Assuming each sequence has a length of 10 and 3 features
X_train = np.random.randn(1000, 10, 3)
y_train = np.random.randint(2, size=1000)  # 0s and 1s
X_test = np.random.randn(350, 10, 3)
y_test = np.random.randint(2, size=350)  # 0s and 1s

# 1. Building the LSTM with Attention Model
sequence_length = 10
num_features = 3

input_layer = Input(shape=(sequence_length, num_features))

# LSTM layers
x = LSTM(50, return_sequences=True)(input_layer)
x = Dropout(0.2)(x)
x = LSTM(50, return_sequences=True)(input_layer)
x = Dropout(0.2)(x)

# Attention layer
#query = LSTM(50, return_sequences=True)(x)
#value = LSTM(50, return_sequences=True)(x)
#x = Attention()([query, value])

x = Flatten()(x)
x = Dense(50, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

model.summary()

# 2. Adaptive Sampling
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_train.reshape(-1, sequence_length * num_features), y_train)
X_resampled = X_resampled.reshape(-1, sequence_length, num_features)

# 3. Training the Model
#history = model.fit(X_resampled, y_resampled, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# 4. Anomaly Detection
predictions = model.predict(X_test)
anomalies = (predictions > 0.3).astype(int)

#history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Make predictions
#y_pred = model.predict(X_test)

# Flatten the predictions and true values for simplicity
#y_true_flat = y_test.flatten()
#y_pred_flat = y_pred.flatten()

# Log model parameters
mlflow.log_param("input_seq_length", input_seq_length)
mlflow.log_param("output_seq_length", output_seq_length)
mlflow.log_param("epochs", 100)
mlflow.log_param("batch_size", 32)

model.summary()

# Training the Model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Logging the metrics
for epoch in range(100):
    mlflow.log_metric("train_loss", history.history['loss'][epoch], step=epoch)
    mlflow.log_metric("train_accuracy", history.history['accuracy'][epoch], step=epoch)
    mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)
    mlflow.log_metric("val_accuracy", history.history['val_accuracy'][epoch], step=epoch)

# Log the model
mlflow.keras.log_model(model, "model")

# End the MLflow experiment
mlflow.end_run()