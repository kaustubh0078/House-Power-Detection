from fastapi import FastAPI
import pandas as pd
from src.data_preprocessing import load_and_process
from src.prophet_model import train_prophet
from src.lstm_model import train_lstm

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Energy Forecasting API is running!"}

@app.get("/forecast/prophet")
def prophet_forecast(days: int = 7):
    processed_path = "data/processed/power_consumption_sample.csv"
    forecast, _ = train_prophet(processed_path, forecast_days=days, save_plot=False)
    return forecast[["ds", "yhat"]].tail(days * 24).to_dict(orient="records")

@app.get("/forecast/lstm")
def lstm_forecast(epochs: int = 10):
    processed_path = "data/processed/power_consumption_sample.csv"
    _, y_test, y_pred = train_lstm(processed_path, window_size=24, epochs=epochs, save_plot=False)
    return {"actual": y_test.flatten().tolist()[:200], "predicted": y_pred.flatten().tolist()[:200]}
