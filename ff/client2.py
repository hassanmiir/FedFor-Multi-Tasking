import pandas as pd
import numpy as np
import tensorflow as tf
import flwr as fl
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Attention
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt


def load_data():
    # Load the dataset
    data = pd.read_csv('/home/disi/ff/Datasets/Dataset_3/Occupancy_Estimation.csv')

    # Assuming 'Date' is a column with date information
    # and another column (e.g., 'Time') has time information
    data['Date'] = pd.to_datetime(data['Date'])
    data['year'] = data['Date'].dt.year
    data['month'] = data['Date'].dt.month
    data['day'] = data['Date'].dt.day
    data.drop('Date', axis=1, inplace=True)

    # We have  there is a time column, process it similarly
    if 'Time' in data.columns:
        data['Time'] = pd.to_datetime(data['Time'], format='%H:%M:%S').dt.time
        data['hour'] = data['Time'].apply(lambda x: x.hour)
        data['minute'] = data['Time'].apply(lambda x: x.minute)
        data['second'] = data['Time'].apply(lambda x: x.second)
        data.drop('Time', axis=1, inplace=True)

    # Rest of the preprocessing
    X = data.iloc[:, :-1].values
    X = np.expand_dims(X, axis=2)
    y = data.iloc[:, -1].values
    X_scaled = (X - X.min(0))/(X.max(0) - X.min(0))
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#Creating model using LSTM as baseline model

def create_model(input_shape):
    # Input layer
    inputs = Input(shape=input_shape)

    # LSTM layer with return_sequences=True
    lstm_out = LSTM(50, return_sequences=True)(inputs)

    # Attention layer
    attention_out = Attention()([lstm_out, lstm_out])

    # Additional layers
    attention_flatten = tf.keras.layers.Flatten()(attention_out)
    dense_out = Dense(50, activation='relu')(attention_flatten)
    outputs = Dense(1, activation='sigmoid')(dense_out)

    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    return model

# Flower client class
class FedForClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights() 

    def fit(self, parameters, config):
        # Set model parameters, train the model, return the updated model parameters
        model.set_weights(parameters)
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
        return model.get_weights(), len(X_train), {}

    def evaluate(self, parameters, config):
        # Set model parameters and evaluate the model
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
        return loss, len(X_val), {"accuracy": accuracy}



    
    # Main script to run the Flower client
if __name__ == "__main__":
    X_train, X_val, y_train, y_val = load_data()
    model = create_model((X_train.shape[1], 1))
    fl.client.start_numpy_client(server_address="localhost:8080", client=FedForClient())
