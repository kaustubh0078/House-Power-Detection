import os
import sys
import pandas as pd
import streamlit as st

# --- Fix imports ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.data_preprocessing import load_and_process
from src.prophet_model import train_prophet
from src.lstm_model import train_lstm
from src.evaluation import evaluate_forecast


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Energy Consumption Forecasting", layout="wide")

st.title("⚡ Energy Consumption Forecasting")
st.markdown("Forecast household/industrial power usage using **Prophet** or **LSTM**.")

# -------------------------------
# Sidebar: Dataset Options
# -------------------------------
st.sidebar.header("Dataset Options")
uploaded_file = st.sidebar.file_uploader("Upload .txt dataset", type=["txt"])

if uploaded_file is not None:
    raw_path = os.path.join("data", "uploaded_dataset.txt")
    with open(raw_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    processed_path = "data/processed/power_consumption_hourly.csv"
    df = load_and_process(raw_path, processed_path, freq="H")
else:
    processed_path = "data/processed/power_consumption_sample.csv"
    if not os.path.exists(processed_path):
        st.error("⚠️ No dataset available. Please upload a file.")
        st.stop()
    df = pd.read_csv(processed_path, parse_dates=["datetime"], index_col="datetime")


st.write("### Processed Data Preview")
st.dataframe(df.head())

# -------------------------------
# App Mode
# -------------------------------
st.sidebar.header("Choose Mode")
mode = st.sidebar.radio("Select Option", ["Single Model Forecast", "Model Comparison"])

# -------------------------------
# Single Model Mode
# -------------------------------
if mode == "Single Model Forecast":
    model_choice = st.sidebar.selectbox("Choose Model", ["Prophet", "LSTM"])
    forecast_days = st.sidebar.slider("Forecast Horizon (days)", 1, 30, 7)
    epochs = st.sidebar.slider("LSTM Training Epochs", 5, 50, 10)

    if st.sidebar.button("Run Forecast"):
        if model_choice == "Prophet":
            forecast, model = train_prophet(processed_path, forecast_days=forecast_days, save_plot=False)
            st.success("✅ Prophet forecast completed!")
            st.line_chart(forecast.set_index("ds")[["yhat"]].tail(forecast_days*24))

            # Evaluation
            df_eval = df.reset_index().rename(columns={"datetime": "ds", "Global_active_power": "y"})
            prophet_eval = forecast[["ds", "yhat"]].merge(df_eval, on="ds", how="inner").dropna()
            metrics = evaluate_forecast(prophet_eval["y"], prophet_eval["yhat"])

        elif model_choice == "LSTM":
            lstm_model, y_test, y_pred = train_lstm(processed_path, window_size=24, epochs=epochs, save_plot=False)
            st.success("✅ LSTM forecast completed!")
            results_df = pd.DataFrame({"Actual": y_test.flatten(), "Predicted": y_pred.flatten()})
            st.line_chart(results_df.head(200))
            metrics = evaluate_forecast(y_test, y_pred)

        st.write("### 📊 Evaluation Metrics")
        st.json(metrics)

# -------------------------------
# Comparison Mode
# -------------------------------
if mode == "Model Comparison":
    forecast_days = st.sidebar.slider("Forecast Horizon (days)", 1, 30, 7)
    epochs = st.sidebar.slider("LSTM Training Epochs", 5, 50, 10)

    if st.sidebar.button("Run Comparison"):
        st.info("Training Prophet...")
        forecast, prophet_model = train_prophet(processed_path, forecast_days=forecast_days, save_plot=False)
        df_eval = df.reset_index().rename(columns={"datetime": "ds", "Global_active_power": "y"})
        prophet_eval = forecast[["ds", "yhat"]].merge(df_eval, on="ds", how="inner").dropna()
        prophet_metrics = evaluate_forecast(prophet_eval["y"], prophet_eval["yhat"])

        st.info("Training LSTM...")
        lstm_model, y_test, y_pred = train_lstm(processed_path, window_size=24, epochs=epochs, save_plot=False)
        lstm_metrics = evaluate_forecast(y_test, y_pred)

        # Layout: Side-by-side
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📈 Prophet Forecast")
            st.line_chart(forecast.set_index("ds")[["yhat"]].tail(forecast_days*24))
            st.write("Metrics:", prophet_metrics)

        with col2:
            st.subheader("📈 LSTM Forecast")
            results_df = pd.DataFrame({"Actual": y_test.flatten(), "Predicted": y_pred.flatten()})
            st.line_chart(results_df.head(200))
            st.write("Metrics:", lstm_metrics)
