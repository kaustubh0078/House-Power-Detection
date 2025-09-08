import os
import sys
import pandas as pd

# --- Fix for imports (make src/ discoverable) ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.evaluation import evaluate_forecast
from src.prophet_model import train_prophet
from src.lstm_model import train_lstm


def compare_models(processed_path: str, forecast_days: int = 7, lstm_epochs: int = 10):
    """
    Train Prophet and LSTM, compare performance on the same dataset.

    Parameters
    ----------
    processed_path : str
        Path to processed dataset.
    forecast_days : int
        Horizon for Prophet forecast (in days).
    lstm_epochs : int
        Training epochs for LSTM.
    """

    # --- Prophet Forecast ---
    print("[INFO] Training Prophet...")
    forecast, prophet_model = train_prophet(processed_path, forecast_days=forecast_days, save_plot=True)

    # Align Prophet forecast with actual values
    df = pd.read_csv(processed_path, parse_dates=["datetime"])
    df.rename(columns={"datetime": "ds", "Global_active_power": "y"}, inplace=True)
    prophet_eval = forecast[["ds", "yhat"]].merge(df, on="ds", how="inner").dropna()
    prophet_metrics = evaluate_forecast(prophet_eval["y"], prophet_eval["yhat"])

    # --- LSTM Forecast ---
    print("[INFO] Training LSTM...")
    lstm_model, y_test, y_pred = train_lstm(processed_path, window_size=24, epochs=lstm_epochs, save_plot=True)
    lstm_metrics = evaluate_forecast(y_test, y_pred)

    # --- Results ---
    print("\n[RESULTS] Prophet vs LSTM")
    print("Prophet:", prophet_metrics)
    print("LSTM:   ", lstm_metrics)

    return {"Prophet": prophet_metrics, "LSTM": lstm_metrics}


if __name__ == "__main__":
    processed_path = os.path.join("data", "processed", "power_consumption_hourly.csv")
    results = compare_models(processed_path, forecast_days=7, lstm_epochs=10)

    # Save metrics to results/metrics.txt
    os.makedirs("results", exist_ok=True)
    with open("results/metrics.txt", "w") as f:
        for model, metrics in results.items():
            f.write(f"{model}:\n")
            for k, v in metrics.items():
                f.write(f"  {k}: {v:.3f}\n")
            f.write("\n")

    print("[INFO] Metrics saved to results/metrics.txt")
