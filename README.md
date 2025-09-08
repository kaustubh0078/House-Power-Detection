# ⚡ Energy Consumption Forecasting

An interactive app to **forecast household or industrial electricity consumption** using **Prophet** (time-series) and **LSTM** (deep learning).
Built with **Python + Streamlit**.

---

## 🚀 Features
- Upload your own `.txt` dataset (in the UCI Household Power Consumption format).
- Use the built-in **sample dataset** for an instant demo.
- Choose between two powerful forecasting models:
  - 📈 **Prophet**: Excellent for analyzing trends and seasonality.
  - 🤖 **LSTM**: A deep learning model for sequential forecasting.
- Evaluate model performance with **RMSE**, **MAE**, and **MAPE** metrics.
- Visualize results with interactive plots.

---

## 📊 Demo
You can try the live application here:

**[⚡ Live App on Streamlit Cloud](https://house-power-detection-gnyuwfzpbsplmcxxgfavs3.streamlit.app/)**

---

## 📦 Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/kaustubh0078/House-Power-Consumption.git](https://github.com/kaustubh0078/House-Power-Consumption.git)
    cd House-Power-Consumption
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    # Create and activate a virtual environment (optional but recommended)
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

    # Install the required packages
    pip install -r requirements.txt
    ```

---

## ▶️ Run Locally

1.  **Run the preprocessing step once** to generate the processed data file:
    ```bash
    python src/data_preprocessing.py
    ```

2.  **Launch the Streamlit app:**
    ```bash
    streamlit run app/streamlit_app.py
    ```
    Your browser should open with the app running locally.

---

## 📈 Models

### Prophet
Prophet is a forecasting procedure developed by Facebook. It excels at capturing **trends**, **seasonality**, and **holiday effects**, making it great for producing interpretable forecasts.

### LSTM
Long Short-Term Memory (LSTM) is a type of recurrent neural network (RNN). It's designed for **sequential data** and is capable of learning complex, long-term temporal dependencies in time-series data.

---

## 📡 Deployment
This application is deployed on **Streamlit Cloud**. The deployment is automatically triggered and updated whenever changes are pushed to the `main` branch of the GitHub repository.

---

## 🧑‍💻 Author
Built by **[Kaustubh Jaiswal](https://github.com/kaustubh0078)** ✨
