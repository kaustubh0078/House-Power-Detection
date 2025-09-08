import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def create_sequences(data, window_size):
    """Generate input/output sequences for LSTM."""
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

def train_lstm(processed_path: str,
               window_size: int = 24,
               forecast_steps: int = 24,
               epochs: int = 20,
               save_plot: bool = True):
    """
    Train an LSTM model for energy consumption forecasting.

    Parameters
    ----------
    processed_path : str
        Path to processed dataset (csv).
    window_size : int
        Number of past hours to use for prediction.
    forecast_steps : int
        How many hours ahead to forecast.
    epochs : int
        Training epochs.
    save_plot : bool
        Whether to save results plot.
    """

    # Load processed data
    df = pd.read_csv(processed_path, parse_dates=["datetime"], index_col="datetime")
    values = df["Global_active_power"].values.reshape(-1, 1)

    # Scale data to [0,1] for LSTM
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    # Create sequences
    X, y = create_sequences(scaled, window_size)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Train/test split (80/20)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build LSTM model
    model = Sequential([
        LSTM(64, activation="relu", input_shape=(window_size, 1)),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")

    # Train
    model.fit(X_train, y_train, epochs=epochs, validation_split=0.2, verbose=1)

    # Predict
    y_pred = model.predict(X_test)

    # Inverse transform predictions and actual
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1))
    y_pred_inv = scaler.inverse_transform(y_pred)

    # Metrics
    rmse = sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    mae = mean_absolute_error(y_test_inv, y_pred_inv)

    print(f"[INFO] LSTM RMSE: {rmse:.3f}, MAE: {mae:.3f}")

    # Plot
    plt.figure(figsize=(15,5))
    plt.plot(y_test_inv[:forecast_steps], label="Actual")
    plt.plot(y_pred_inv[:forecast_steps], label="Predicted")
    plt.title("LSTM Forecast (Test Sample)")
    plt.xlabel("Hours")
    plt.ylabel("kW")
    plt.legend()

    if save_plot:
        os.makedirs("results", exist_ok=True)
        plt.savefig("results/lstm_forecast.png")

    return model, y_test_inv, y_pred_inv


if __name__ == "__main__":
    processed_path = "data/processed/power_consumption_hourly.csv"
    model, actual, predicted = train_lstm(processed_path, window_size=24, epochs=10)
