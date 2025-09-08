import pandas as pd
import os

def load_and_process(input_path: str, output_path: str, freq: str = "H"):
    """
    Load raw household power consumption data, clean, resample, and save processed file.

    Parameters
    ----------
    input_path : str
        Path to raw .txt file (semicolon separated).
    output_path : str
        Path to save processed .csv file.
    freq : str
        Resampling frequency: 'H' = hourly, 'D' = daily, etc.
    """

    # Load dataset
    df = pd.read_csv(
        input_path,
        sep=";",
        na_values=["?"],    # missing values are represented as "?"
        low_memory=False
    )

    # Combine Date + Time into datetime
    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce"
    )

    # Drop unused columns
    df.drop(columns=["Date", "Time"], inplace=True)

    # Set datetime as index
    df.set_index("datetime", inplace=True)

    # Convert target variable to numeric
    df["Global_active_power"] = pd.to_numeric(df["Global_active_power"], errors="coerce")

    # Resample to desired frequency (default: hourly mean)
    df_resampled = df["Global_active_power"].resample(freq).mean().dropna()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save processed data
    df_resampled.to_csv(output_path)

    print(f"[INFO] Processed dataset saved at: {output_path}")
    print(df_resampled.head())

    return df_resampled


if __name__ == "__main__":
    raw_path = "data/household_power_consumption.txt"
    processed_path = "data/processed/power_consumption_hourly.csv"

    load_and_process(raw_path, processed_path, freq="H")
