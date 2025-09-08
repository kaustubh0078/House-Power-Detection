import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os

def train_prophet(processed_path: str, forecast_days: int = 30, save_plot: bool = True):
    """
    Train Prophet model on energy consumption and forecast future usage.

    Parameters
    ----------
    processed_path : str
        Path to processed dataset (csv).
    forecast_days : int
        Number of days to forecast ahead.
    save_plot : bool
        Whether to save forecast plot in results/.
    """

    # Load processed data
    df = pd.read_csv(processed_path, parse_dates=["datetime"])
    df.rename(columns={"datetime": "ds", "Global_active_power": "y"}, inplace=True)

    # Fit Prophet model
    model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    model.fit(df)

    # Create future dataframe
    future = model.make_future_dataframe(periods=forecast_days * 24, freq="H")  # hourly forecast
    forecast = model.predict(future)

    # Plot results
    fig1 = model.plot(forecast)
    plt.title(f"Prophet Forecast - Next {forecast_days} Days")

    # Save plot
    if save_plot:
        os.makedirs("results", exist_ok=True)
        fig1.savefig(f"results/prophet_forecast.png")

    # Plot seasonality components
    fig2 = model.plot_components(forecast)
    if save_plot:
        fig2.savefig(f"results/prophet_components.png")

    print("[INFO] Prophet forecast completed.")
    return forecast, model


if __name__ == "__main__":
    processed_path = "data/processed/power_consumption_hourly.csv"
    forecast, model = train_prophet(processed_path, forecast_days=7)
    print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail())
