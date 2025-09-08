import pandas as pd

# Path to your full processed hourly file (local)
input_path = "data/processed/power_consumption_hourly.csv"

# Load full processed dataset
df = pd.read_csv(input_path, parse_dates=["datetime"], index_col="datetime")

# Take only the first 7 days (or adjust as needed)
sample = df.head(24*7)  # first 168 hours

# Save sample
sample_path = "data/processed/power_consumption_sample.csv"
sample.to_csv(sample_path)
print(f"[INFO] Sample dataset saved to {sample_path}, shape = {sample.shape}")
