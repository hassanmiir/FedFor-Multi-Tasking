# Import Libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_score, recall_score, f1_score, accuracy_score

# File paths
train_file_1 = '/home/hassan/FedFor/FF/Datasets/Dataset_2/Train1.csv'
train_file_2 = '/home/hassan/FedFor/FF/Datasets/Dataset_2/Train2.csv'
test_file = '/home/hassan/FedFor/FF/Datasets/Dataset_2/Test.csv'

# Load datasets
train_data_1 = pd.read_csv(train_file_1)
train_data_2 = pd.read_csv(train_file_2)
test_data = pd.read_csv(test_file)

train_data = pd.concat([train_data_1, train_data_2], ignore_index=True)
if 'Date' in train_data.columns:
    train_data['Date'] = pd.to_datetime(train_data['Date'])
    train_data = train_data.sort_values(by='Date')
if 'Date' in test_data.columns:
    test_data['Date'] = pd.to_datetime(test_data['Date'])
    test_data = test_data.sort_values(by='Date')

# Selecting relevant columns
col_names = ['Temperature', 'Humidity', 'Light', 'CO2', 'Occupancy']
train_forecast = train_data[col_names]
test_forecast = test_data[col_names]

# Scale data
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_forecast)
test_scaled = scaler.transform(test_forecast)

# Create sequences
def create_sequences(data, input_seq_length, output_seq_length):
    X, y_forecast, y_occupancy = [], [], []
    for i in range(len(data) - input_seq_length - output_seq_length + 1):
        seq_x = data[i:i + input_seq_length, :4]
        seq_y_forecast = data[i + input_seq_length:i + input_seq_length + output_seq_length, 0]  # Forecast target
        seq_y_occupancy = data[i:i + input_seq_length, 4]
        X.append(seq_x)
        y_forecast.append(seq_y_forecast)
        y_occupancy.append(seq_y_occupancy)
    return np.array(X), np.array(y_forecast), np.array(y_occupancy)

input_seq_length = 10
output_seq_length = 3
features = 4

X_train, y_train_forecast, y_train_occupancy = create_sequences(train_scaled, input_seq_length, output_seq_length)
X_test, y_test_forecast, y_test_occupancy = create_sequences(test_scaled, input_seq_length, output_seq_length)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train_forecast = torch.tensor(y_train_forecast, dtype=torch.float32)
y_train_occupancy = torch.tensor(y_train_occupancy, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test_forecast = torch.tensor(y_test_forecast, dtype=torch.float32)
y_test_occupancy = torch.tensor(y_test_occupancy, dtype=torch.float32)

train_dataset = TensorDataset(X_train, y_train_forecast, y_train_occupancy)
test_dataset = TensorDataset(X_test, y_test_forecast, y_test_occupancy)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Device management
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define Attention Mechanism
class Attention(nn.Module):
    """
    Implements an attention mechanism to compute weighted temporal summaries.
    """
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.attention_weights = nn.Linear(input_dim, 1)

    def forward(self, x):
        # Compute attention scores
        weights = self.attention_weights(x)  # Shape: [batch_size, seq_length, 1]
        weights = torch.softmax(weights, dim=1)  # Normalize across sequence
        # Compute weighted sum across sequence
        attended_output = torch.sum(x * weights, dim=1)  # Shape: [batch_size, input_dim]
        return attended_output, weights

# Define Model with Task-Specific and Cross-Task Attention
class DualTaskLSTMWithAttention(nn.Module):
    """
    Dual-task LSTM model with task-specific attention and cross-task attention.
    """
    def __init__(self, input_size, hidden_size_shared, hidden_size_task, output_size):
        super(DualTaskLSTMWithAttention, self).__init__()

        # Shared LSTM for encoding the input sequence
        self.shared_lstm = nn.LSTM(input_size, hidden_size_shared, batch_first=True)

        # Task-Specific Attention
        self.forecasting_attention = Attention(hidden_size_shared)
        self.occupancy_attention = Attention(hidden_size_shared)

        # Forecasting task-specific layers
        self.forecasting_lstm = nn.LSTM(hidden_size_shared, hidden_size_task, batch_first=True)
        self.forecasting_out = nn.Linear(hidden_size_task, output_size)

        # Occupancy detection task-specific layers
        self.occupancy_lstm = nn.LSTM(hidden_size_shared, hidden_size_task, batch_first=True)
        self.occupancy_out = nn.Linear(hidden_size_task, 1)
        self.sigmoid = nn.Sigmoid()

        # Cross-Task Attention Mechanism
        self.cross_task_attention = nn.Linear(2 * hidden_size_task, hidden_size_task)

    def forward(self, x):
        # Shared LSTM encodes the input sequence
        shared_out, _ = self.shared_lstm(x)  # Shape: [batch_size, seq_length, hidden_size_shared]

        # Task-Specific Attention
        forecast_attended, _ = self.forecasting_attention(shared_out)  # Shape: [batch_size, hidden_size_shared]
        occupancy_attended, _ = self.occupancy_attention(shared_out)  # Shape: [batch_size, hidden_size_shared]

        # Task-Specific Outputs
        # Forecasting Branch
        forecast_out, _ = self.forecasting_lstm(forecast_attended.unsqueeze(1))  # Add sequence dimension back
        forecast_out = self.forecasting_out(forecast_out[:, -1, :])  # Use the last time step for forecasting

        # Occupancy Detection Branch
        occupancy_out, _ = self.occupancy_lstm(occupancy_attended.unsqueeze(1))  # Add sequence dimension back
        occupancy_out = self.occupancy_out(occupancy_out[:, -1, :])  # Use the last time step for occupancy
        occupancy_out = self.sigmoid(occupancy_out)  # Convert to probabilities

        # Cross-Task Attention
        cross_task_input = torch.cat([forecast_attended, occupancy_attended], dim=1)  # Concatenate task outputs
        cross_task_out = torch.tanh(self.cross_task_attention(cross_task_input))  # Integrate across tasks

        return forecast_out, occupancy_out, cross_task_out

# Initialize model
model = DualTaskLSTMWithAttention(input_size=features, hidden_size_shared=32, hidden_size_task=32, output_size=output_seq_length).to(device)


# Training loop
num_epochs = 50
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-6)
criterion_forecasting = nn.MSELoss()
criterion_occupancy = nn.BCELoss()

for epoch in range(num_epochs):
    model.train()
    total_loss_forecasting = 0
    total_loss_occupancy = 0
    for X_batch, y_forecast_batch, y_occupancy_batch in train_loader:
        # Move data to device
        X_batch = X_batch.to(device)
        y_forecast_batch = y_forecast_batch.to(device)
        y_occupancy_batch = y_occupancy_batch.to(device)

        # Forward pass
        forecast_out, occupancy_out, cross_task_out = model(X_batch)

        # Compute losses
        loss_forecasting = criterion_forecasting(forecast_out, y_forecast_batch)
        loss_occupancy = criterion_occupancy(occupancy_out.squeeze(), y_occupancy_batch[:, -1])
        loss = loss_forecasting + loss_occupancy  # Total loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate losses for logging
        total_loss_forecasting += loss_forecasting.item()
        total_loss_occupancy += loss_occupancy.item()

    print(f"Epoch {epoch+1}, Forecasting Loss: {total_loss_forecasting:.4f}, Occupancy Loss: {total_loss_occupancy:.4f}")

# Evaluation
model.eval()
y_true_forecasting, y_pred_forecasting, y_true_occupancy, y_pred_occupancy = [], [], [], []

with torch.no_grad():
    for X_batch, y_forecast_batch, y_occupancy_batch in test_loader:
        X_batch = X_batch.to(device)
        forecast_out, occupancy_out, _ = model(X_batch)

        # Forecasting results
        y_true_forecasting.extend(y_forecast_batch.cpu().numpy())
        y_pred_forecasting.extend(forecast_out.cpu().numpy())

        # Occupancy results
        y_true_occupancy.extend(y_occupancy_batch[:, -1].cpu().numpy())
        y_pred_occupancy.extend(occupancy_out.squeeze().cpu().numpy())

# Convert occupancy predictions to binary
y_true_occupancy = np.array(y_true_occupancy).astype(int)
y_pred_occupancy = np.round(y_pred_occupancy).astype(int)

# Metrics
mae_forecasting = mean_absolute_error(y_true_forecasting, y_pred_forecasting)
rmse_forecasting = np.sqrt(mean_squared_error(y_true_forecasting, y_pred_forecasting))
accuracy_occupancy = accuracy_score(y_true_occupancy, y_pred_occupancy)
precision_occupancy = precision_score(y_true_occupancy, y_pred_occupancy)
recall_occupancy = recall_score(y_true_occupancy, y_pred_occupancy)
f1_occupancy = f1_score(y_true_occupancy, y_pred_occupancy)

print(f"Forecasting Metrics - MAE: {mae_forecasting}, RMSE: {rmse_forecasting}")
print(f"Occupancy Detection Metrics - Accuracy: {accuracy_occupancy}, Precision: {precision_occupancy}, Recall: {recall_occupancy}, F1 Score: {f1_occupancy}")
