# Databricks notebook source
"""
Initial Data Ingestion Pipeline

Team: Garden of Eden
Dataset: Kaggle Playground Series S5E6 - Predicting Optimal Fertilizers
Author(s): David Goh, Arkojit Ghosh, Yu-Hua Chen, Aliza Tarakova

Objective: Process raw fertilizer prediction data and load it into Databricks tables 
for ML pipeline consumption.
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

from pathlib import Path
from pyspark.sql import SparkSession
import pandas as pd

from src.config_manager import get_config
from scripts.initial_data_handling import prepare_initial_datasets

# COMMAND ----------

# Configuration Loading
# Load configuration parameters and set up data paths and processing parameters

config = get_config(base_path='config/config_base.yaml')
RAW_DATA_PATH = Path(config.data.local_raw_path)
TEST_SET_OUTPUT_PATH = Path(config.data.local_test_data_path)
INITIAL_DATASET_OUTPUT_PATH = Path(config.data.local_train_data_path)
TARGET_COLUMN = config.target_column
SEED = config.random_seed
TEST_SAMPLE_SIZE = 150_000
INITIAL_SAMPLE_SIZE = 150_000
EXCLUDE_LABEL = 'DAP'

print(f"Raw data path: {RAW_DATA_PATH}")
print(f"Test output path: {TEST_SET_OUTPUT_PATH}")
print(f"Initial dataset output path: {INITIAL_DATASET_OUTPUT_PATH}")
print(f"Target column: {TARGET_COLUMN}")

# COMMAND ----------

# Initial Dataset Preparation
# Process raw data and create initial train/test splits
# Apply initial data cleaning and sampling

prepare_initial_datasets(
    raw_path=RAW_DATA_PATH,
    test_output=TEST_SET_OUTPUT_PATH,
    initial_output=INITIAL_DATASET_OUTPUT_PATH,
    target_col=TARGET_COLUMN,
    test_size=TEST_SAMPLE_SIZE,
    initial_size=INITIAL_SAMPLE_SIZE,
    exclude_label=EXCLUDE_LABEL,
    random_seed=SEED
)

print("Initial datasets prepared successfully")

# COMMAND ----------

# Save Data to Databricks Tables
# Load processed data into Spark DataFrames
# Save to Databricks tables for downstream consumption
# Create raw, intermediate, and features tables

# Configure table names from config
raw_table = f'{config.data.catalog_name}.{config.data.schema_name}.{config.data.raw_table_name}'
intermediate_table = f'{config.data.catalog_name}.{config.data.schema_name}.{config.data.intermediate_clean_table}'
features_table = f'{config.data.catalog_name}.{config.data.schema_name}.{config.data.features_table_name}'

# Initialize Spark session
spark = SparkSession.builder.getOrCreate()

# Load and save raw data
df_raw_pd = pd.read_csv(os.path.join(os.getcwd(), RAW_DATA_PATH))
df_raw = spark.createDataFrame(df_raw_pd)
df_raw.write.mode("overwrite").saveAsTable(raw_table)
print(f"Raw data saved to table: {raw_table}")

# Load and save initial cleaned data
df_initial = pd.read_csv(os.path.join(os.getcwd(), INITIAL_DATASET_OUTPUT_PATH))
df_initial = spark.createDataFrame(df_initial)
df_initial.write.mode("overwrite").saveAsTable(intermediate_table)
print(f"Initial cleaned data saved to table: {intermediate_table}")

# Create and save features table
df_features = df_initial.select(config.data.features.numerical + config.data.features.categorical + [TARGET_COLUMN])
df_features.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(features_table)
print(f"Feature data saved to table: {features_table}")

print("Data ingestion pipeline completed successfully")