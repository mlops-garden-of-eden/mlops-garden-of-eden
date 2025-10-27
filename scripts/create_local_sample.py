import pandas as pd
from pathlib import Path

RAW_DATA_PATH = Path("data/train.csv")
LOCAL_SAMPLE_PATH = Path("data/dataset_small.csv")

df = pd.read_csv(RAW_DATA_PATH)
sample = df.sample(n=100, random_state=42)
LOCAL_SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
sample.to_csv(LOCAL_SAMPLE_PATH, index=False)
print(f"Saved local sample ({len(sample)} rows) to {LOCAL_SAMPLE_PATH}")