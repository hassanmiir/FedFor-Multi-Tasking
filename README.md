A Federated Multi-Task Learning Framework with Dual Attention Mechanisms for Smart Buildings
2025 IEEE 101st Vehicular Technology Conference (VTC2025-Spring), Oslo, Norway

Paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=11174387

Repository: https://github.com/hassanmiir/FedFor-Multi-Tasking

Overview

This work proposes  a federated multi-task learning framework with dual attention mechanisms for smart building applications.
DualFL integrates:

Feature-level attention to identify relevant sensor signals

Task-level attention to capture dependencies across forecasting tasks

Federated learning to preserve privacy by training models locally without sharing raw data

Applications include temperature forecasting, environmental monitoring, and distributed IoT systems in smart buildings.

The repository also includes baseline implementations with single datasets and standard attention mechanisms for comparative analysis.

Repository Structure

FedFor-Multi-Tasking/
├─ Datasets/ # Place raw/processed datasets here
├─ Fedfor_LSTM/ # LSTM-focused experiments
├─ ff_lstm/ # Additional LSTM experiments
├─ mlruns/ # MLflow tracking (auto-created)
├─ mlartifacts/.../artifacts/ # MLflow artifacts (models, metrics)
│
├─ dualfl.py # Main DualFL implementation (federated + dual attention)
├─ gru_attention.py # GRU + attention baseline
├─ lstm_with_attention.py # LSTM + attention baseline
├─ lstm_attention_d1.py # LSTM + Attention (Dataset D1)
├─ lstm_attention_d1_final.py # Finalized LSTM + Attention (D1)
├─ Fedfor_GRU_Attention_D1.py # GRU + Attention (D1)
├─ Fedfor_GRU_Attention_D1_c.py # Corrected variant of GRU + Attention (D1)
├─ Fedfor_GRU_D1.py # GRU (Dataset D1)
├─ Fedfor_GRU_D2.py # GRU (Dataset D2)
├─ Fedfor_LSTM_D1.py # LSTM (Dataset D1)
├─ Fedfor_LSTM_D2.py # LSTM (Dataset D2)
│
├─ Forecasting.ipynb # Notebook for forecasting experiments
├─ GRU_attention_d2.ipynb # GRU + Attention (D2) notebook
├─ Lstm_attention_d2.ipynb # LSTM + Attention (D2) notebook
├─ LSTM.ipynb # LSTM-only experiments
├─ lstm_attention_d1_final.ipynb # Notebook for LSTM + Attention (D1 final)
│
├─ lstm_attention_model.h5 # Example saved model
├─ scaler.pkl # Example saved scaler
├─ requirements.txt # Python dependencies
└─ README.md # Project documentation

Installation

Requirements:

Python 3.8+

PyTorch >= 1.10

NumPy, Pandas, Matplotlib

PyYAML

MLflow (optional, for experiment tracking)

Install dependencies:

pip install -r requirements.txt


To enable MLflow experiment tracking:

pip install mlflow
mlflow ui   # Browse at http://127.0.0.1:5000

Datasets

Place your sensor datasets inside the Datasets/ folder.
Scripts refer to D1 and D2 datasets. Make sure your files match those names or update the paths in each script.
Typical datasets: temperature sensors, environmental monitoring, time-series forecasting for smart buildings.


python dualfl.py





Jupyter notebooks
Run experiments interactively:

Forecasting.ipynb

GRU_attention_d2.ipynb

Lstm_attention_d2.ipynb

LSTM.ipynb

lstm_attention_d1_final.ipynb

MLflow logs and models will appear under mlruns/ automatically.

Results

DualFL consistently achieves higher accuracy and better generalization than single-task baselines.
The dual attention mechanism helps capture both sensor-level and task-level dependencies.
See the paper for detailed results, ablation studies, and analysis: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=11174387

Citation

Please cite if you use this code:

@INPROCEEDINGS{11174387,
  author={Hassan, Mir and Yildrim, Kasim Sinan and Iacca, Giovanni},
  booktitle={2025 IEEE 101st Vehicular Technology Conference (VTC2025-Spring)}, 
  title={A Federated Multi-Task Learning Framework with Dual Attention Mechanisms for Smart Buildings}, 
  year={2025},
  pages={1-7},
  keywords={Temperature sensors;Data privacy;Vehicular and wireless technologies;Temperature distribution;Attention mechanisms;Accuracy;Smart buildings;Federated learning;Multitasking;Forecasting;Federated learning;Time-Series;Environmental;Monitoring;Distributed Systems;Sensors},
  doi={10.1109/VTC2025-Spring65109.2025.11174387}}

Contact

For questions and collaborations:
Mir Hassan – meerrhassan@gmail.com
