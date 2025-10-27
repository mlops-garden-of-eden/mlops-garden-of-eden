# Databricks notebook source
"""
Data Ingestion Pipeline Overview

Team: Garden of Eden
Dataset: Kaggle Playground Series S5E6 - Fertilizer Optimization
Authors: David Goh, Arkojit Ghosh, Yu-Hua Chen, Aliza Tarakova

Purpose: Ingest and process raw fertilizer prediction data, loading it into Databricks tables for downstream machine learning workflows.
"""

# COMMAND ----------

# Environment Setup
# Ensure correct working directory and environment
# Verify imports work properly

import os

# Get current notebook's directory
notebook_dir = os.path.dirname(os.path.realpath(__file__)) if '__file__' in dir() else os.getcwd()

# Navigate up to BASE
base_dir = os.path.dirname(notebook_dir)  # Go up one level from notebooks/
os.chdir(base_dir)

print(f"Working directory: {os.getcwd()}")

# COMMAND ----------

# Import Required Libraries
# Import data processing libraries and custom modules from src/
import pandas as pd

from src.data_loader import load_training_data
from src.config_manager import get_config

# COMMAND ----------

config = get_config(base_path='config/config_base.yaml')
df_local = load_training_data(config.data, source="local")
print(df_local.head())

# Example usage for Databricks loading
df_db = load_training_data(config.data, source="databricks")
print(df_db.head())


# COMMAND ----------

# Check if the data are the same after sorted by id
df_local = df_local.sort_values(by='id').reset_index(drop=True)
df_db = df_db.sort_values(by='id').reset_index(drop=True)
pd.testing.assert_frame_equal(df_local, df_db)