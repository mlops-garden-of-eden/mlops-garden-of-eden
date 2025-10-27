"""Data ingestion script for MLOps Garden of Eden project.

This module provides functionality to ingest data from various sources,
clean and preprocess the data, and store it in a data lake following
the bronze-silver-gold architecture pattern.
"""

import argparse
import os
import sys
import uuid
from typing import Optional

import gdown
import pandas as pd
import requests
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Get base path from environment variable or use default
BASE_PATH = os.environ.get(
    "BASE_PATH", "/Workspace/Users/chenjoachim@cs.toronto.edu/mlops-garden-of-eden"
)
sys.path.insert(0, BASE_PATH)

from src.config_manager import Config, DataConfig, get_config


def clean_data(config: Config, df: pd.DataFrame) -> pd.DataFrame:
    """Clean the input DataFrame by handling missing values and standardizing column names.

    Args:
        config: The configuration object containing cleaning settings.
        df: The input DataFrame to be cleaned.

    Returns:
        The cleaned DataFrame.
    """
    # Standardize column names
    df.columns = [
        col.strip().lower().replace(" ", "_").replace("temparature", "temperature")
        for col in df.columns
    ]

    # Keep only: id, features, target (NOT metadata like _source,
    # _ingestion_timestamp)
    keep_cols = (
        ["id"]
        + config.data.features.numerical
        + config.data.features.categorical
        + [config.target_column]
    )
    for col in df.columns:
        if col not in keep_cols:
            df = df.drop(columns=[col])

    df = df.dropna()  # Drop rows with any missing values

    return df


def feature_engineer_data(config: Config, df: pd.DataFrame) -> pd.DataFrame:
    """Perform feature engineering on the input DataFrame.

    Args:
        config: The configuration object containing feature settings.
        df: The input DataFrame to be feature engineered.

    Returns:
        The DataFrame with engineered features.
    """
    # Placeholder for feature engineering logic
    return df


def ingest_data(config: Config, file_path: str, source: str = "local") -> None:
    """Ingest data from a specified file path into a Spark DataFrame.

    This function implements a bronze-silver-gold data architecture:
    - Bronze: Raw data with metadata
    - Silver: Cleaned data with metadata
    - Gold: Feature-engineered data ready for ML

    Args:
        config: The configuration object containing data settings.
        file_path: Path to the data file to be ingested.
        source: Source type of the data (default: "local").
    """
    data_config = config.data
    spark = SparkSession.builder.appName("DataIngestion").getOrCreate()
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    df.columns = [
        col.strip().lower().replace(" ", "_").replace("temparature", "temperature")
        for col in df.columns
    ]

    # === BRONZE: Raw table with metadata ===
    ingested_raw_df = spark.createDataFrame(df)
    ingested_raw_df = ingested_raw_df.withColumn("_source", F.lit(source)).withColumn(
        "_ingestion_timestamp", F.current_timestamp()
    )

    try:
        raw_df = spark.read.table(
            f"{data_config.catalog_name}.{data_config.schema_name}."
            f"{data_config.raw_table_name}"
        )
        ingested_raw_df.write.mode("append").option("mergeSchema", "true").saveAsTable(
            f"{data_config.catalog_name}.{data_config.schema_name}."
            f"{data_config.raw_table_name}"
        )
    except Exception:
        ingested_raw_df.write.mode("overwrite").option(
            "mergeSchema", "true"
        ).saveAsTable(
            f"{data_config.catalog_name}.{data_config.schema_name}."
            f"{data_config.raw_table_name}"
        )

    # === SILVER: Clean table with metadata ===
    ingested_cleaned_df = spark.createDataFrame(clean_data(config, df))
    ingested_cleaned_df = ingested_cleaned_df.withColumn(
        "_source", F.lit(source)
    ).withColumn("_ingestion_timestamp", F.current_timestamp())

    # Silver keeps: meta_features (id, _source, _ingestion_timestamp) +
    # features + target
    silver_keep_cols = (
        data_config.meta_features
        + data_config.features.numerical
        + data_config.features.categorical
        + [config.target_column]
    )

    try:
        cleaned_df = spark.read.table(
            f"{data_config.catalog_name}.{data_config.schema_name}."
            f"{data_config.intermediate_clean_table}"
        )

        # Drop columns NOT in keep list
        for col in cleaned_df.columns:
            if col not in silver_keep_cols:
                cleaned_df = cleaned_df.drop(col)

        cleaned_df = cleaned_df.dropDuplicates(["id"])
        cleaned_df.write.mode("overwrite").option("mergeSchema", "true").saveAsTable(
            f"{data_config.catalog_name}.{data_config.schema_name}."
            f"{data_config.intermediate_clean_table}"
        )
    except Exception:
        ingested_cleaned_df.write.mode("overwrite").option(
            "mergeSchema", "true"
        ).saveAsTable(
            f"{data_config.catalog_name}.{data_config.schema_name}."
            f"{data_config.intermediate_clean_table}"
        )

    # === GOLD: Feature table (only id + features + target,
    # no other metadata) ===
    feature_df = feature_engineer_data(config, df)
    ingested_featured_df = spark.createDataFrame(feature_df)

    # Gold keeps: id + features + target (no _source, _ingestion_timestamp)
    gold_keep_cols = (
        ["id"]
        + data_config.features.numerical
        + data_config.features.categorical
        + [config.target_column]
    )

    try:
        featured_df = spark.read.table(
            f"{data_config.catalog_name}.{data_config.schema_name}."
            f"{data_config.features_table_name}"
        )

        for col in featured_df.columns:
            if col not in gold_keep_cols:
                featured_df = featured_df.drop(col)

        featured_df = featured_df.dropDuplicates(["id"])
        featured_df.write.mode("overwrite").option("mergeSchema", "true").saveAsTable(
            f"{data_config.catalog_name}.{data_config.schema_name}."
            f"{data_config.features_table_name}"
        )
    except Exception:
        ingested_featured_df.write.mode("overwrite").option(
            "mergeSchema", "true"
        ).saveAsTable(
            f"{data_config.catalog_name}.{data_config.schema_name}."
            f"{data_config.features_table_name}"
        )


def download_file(url: str, save_path: str) -> bool:
    """Download file from a specified URL and save it to a local path.

    Supports both Google Drive URLs and standard HTTP URLs.

    Args:
        url: The URL to download the data from.
        save_path: The local file path to save the downloaded data.

    Returns:
        True if download was successful, False otherwise.
    """
    try:
        if "drive.google.com" in url:
            file_id = url.split("/")[-2]
            gdown.download(
                f"https://drive.google.com/uc?id={file_id}", save_path, quiet=False
            )
            return True
        else:
            response = requests.get(url)
            with open(save_path, "wb") as file:
                file.write(response.content)
            return True
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Data Ingestion Script")
    parser.add_argument(
        "--file_path", type=str, required=True, help="URL or path of the data source"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="local",
        help="Source type of the data. Will attempt to download if not local.",
    )
    return parser.parse_args()


def main() -> None:
    """Main function to orchestrate the data ingestion process."""
    args = parse_args()
    config = get_config(os.path.join(BASE_PATH, "config/config_base.yaml"))
    data_url = args.file_path

    if data_url.startswith("http"):
        # If the data URL is an HTTP URL, download it
        os.makedirs(os.path.join(BASE_PATH, "data"), exist_ok=True)
        tmp_path = os.path.join(BASE_PATH, "data", f"{uuid.uuid4().hex}.csv")
        if download_file(data_url, tmp_path):
            ingest_data(config, tmp_path, source=args.source)
            os.remove(tmp_path)
        else:
            print("Failed to download the data file.")
    else:
        if not os.path.isabs(data_url):
            data_url = os.path.join(BASE_PATH, data_url)
        if os.path.exists(data_url):
            ingest_data(config, data_url, source=args.source)
        else:
            print("Local data file does not exist.")


if __name__ == "__main__":
    main()
