import yaml
import pandas as pd
from pathlib import Path
from src.config_manager import get_config
from src.predictor import run_prediction
import streamlit as st
import os

# Use production config for deployed app, dev for local testing
DEFAULT_CONFIG_PATH = os.getenv("CONFIG_PATH", "config/config_prod.yaml")

# Load config and get feature names
def get_feature_lists(config_path=DEFAULT_CONFIG_PATH):
    config = get_config(config_path)
    num = config.data.features.numerical
    cat = config.data.features.categorical
    # Get unique categorical values from dataset_small.csv
    df = pd.read_csv("data/dataset_small.csv")
    # Use renamed columns if present in config
    rename_map = getattr(config.data, "rename_columns", {}) or {}
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    cat_options = {}
    for col in cat:
        if col in df.columns:
            cat_options[col] = sorted(df[col].dropna().unique())
        else:
            cat_options[col] = []
    return num, cat, cat_options, config

def main():
    st.title("Garden of Eden Fertilizer Predictor")
    st.write("Enter the required features to predict the fertilizer.")

    num_features, cat_features, cat_options, config = get_feature_lists()

    # Build form
    with st.form("prediction_form"):
        num_inputs = {}
        for col in num_features:
            num_inputs[col] = st.number_input(f"{col}", value=0.0, format="%f")
        cat_inputs = {}
        for col in cat_features:
            options = cat_options.get(col, [])
            if options:
                cat_inputs[col] = st.selectbox(f"{col}", options)
            else:
                cat_inputs[col] = st.text_input(f"{col}")
        submitted = st.form_submit_button("Predict Fertilizer")

    if submitted:
        # Prepare input for prediction, ensure 'id' is never sent
        input_dict = {**num_inputs, **cat_inputs}
        input_dict.pop("id", None)
        # Run prediction
        try:
            import numpy as np
            result = run_prediction(config, input_dict)
            fert = result.get("prediction") if isinstance(result, dict) else result["prediction"].iloc[0]
            st.success(f"Predicted Fertilizer: {fert}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

if __name__ == "__main__":
    main()
