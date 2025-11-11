import pandas as pd
from enum import Enum
from pathlib import Path
from typing import Literal

from .config_manager import DataConfig
from .utils import logger
import warnings

class DataSource(Enum):
    """Defines the available data sources for the pipeline."""
    LOCAL = "local"
    DATABRICKS = "databricks"

def load_training_data(
    data_config: DataConfig,
    source: Literal["local", "databricks"]
) -> pd.DataFrame:
    """
    Loads the training dataset either from a local file or a specified data store.
    
    Args:
        data_config (DataConfig): The validated configuration object for data.
        source (Literal["local", "databricks"]): The desired data source.
                                    
    Returns:
        pd.DataFrame: The loaded raw data.
    """
    
    if source == DataSource.LOCAL.value:

        local_path = Path(data_config.local_train_data_path)
        print(f"Loading data locally from: {local_path}")
        
        if not local_path.exists():
            raise FileNotFoundError(f"Local training data file not found at: {local_path}")
            
        df = pd.read_csv(local_path)

        # Apply optional renaming mapping supplied in the config
        rename_map = getattr(data_config, "rename_columns", {}) or {}
        if rename_map:
            # Check for duplicates after rename
            projected = [rename_map.get(c, c) for c in df.columns]
            if len(set(projected)) != len(projected):
                raise ValueError("Column rename mapping would create duplicate column names")

            # Warn for keys not present
            missing = [k for k in rename_map.keys() if k not in df.columns]
            if missing:
                logger.warning(f"rename_columns contains keys not found in CSV: {missing}")

            # Only rename keys that exist in dataframe
            df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        df = df.reset_index(drop=True)
        print(f"Successfully loaded {len(df)} rows from local path.")
        return df

    elif source == DataSource.DATABRICKS.value:
        # --- Databricks/Delta Lake Loading Logic will go here ---
        table_path = f"{data_config.catalog_name}.{data_config.schema_name}.{data_config.intermediate_clean_table}"
        print(f"Loading data from Databricks table: {table_path}")
        # Import PySpark lazily so local environments without pyspark don't fail on module import
        try:
            from pyspark.sql import SparkSession
        except Exception as e:
            raise ImportError(
                "PySpark is required to load data from Databricks but is not installed or failed to import. "
                "If you are running locally and don't need Spark, set data_source: 'local' in your config. "
                "To run Databricks integration locally install pyspark (see requirements-spark.txt) or run on Databricks. "
                f"Original error: {e}"
            )

        spark = SparkSession.builder.appName("DataLoader").getOrCreate()

        table = spark.table(table_path)

        # Apply optional renaming mapping in Spark before converting to pandas
        rename_map = getattr(data_config, "rename_columns", {}) or {}
        if rename_map:
            # Check for duplicates after rename using Spark table column list
            spark_cols = table.columns
            projected = [rename_map.get(c, c) for c in spark_cols]
            if len(set(projected)) != len(projected):
                raise ValueError("Column rename mapping would create duplicate column names (Databricks)")

            for old, new in rename_map.items():
                if old in spark_cols:
                    if old != new:
                        table = table.withColumnRenamed(old, new)
                else:
                    logger.warning(f"rename_columns contains key not found in Databricks table: {old}")

        df = table.toPandas().reset_index(drop=True)

        print(f"Successfully loaded {len(df)} rows from Databricks table.")
        return df
    else:
        raise ValueError(f"Invalid source '{source}'. Must be 'local' or 'databricks'.")
    

def load_prediction_data(
    data_config: DataConfig,
    source: Literal["local", "databricks"]
) -> pd.DataFrame:
    """
    Loads the prediction dataset either from a local file or a specified data store.
    
    Args:
        data_config (DataConfig): The validated configuration object for data.
        source (Literal["local", "databricks"]): The desired data source.
                                    
    Returns:
        pd.DataFrame: The loaded prediction data.
    """
    
    if source == DataSource.LOCAL.value:

        # Use the configured local test path
        local_path = Path(getattr(data_config, 'local_prediction_data_path', '') or data_config.local_test_data_path)
        print(f"Loading prediction data locally from: {local_path}")

        if not local_path.exists():
            raise FileNotFoundError(f"Local prediction data file not found at: {local_path}")
            
        df = pd.read_csv(local_path)

        # Apply optional renaming mapping supplied in the config (same as training loader)
        rename_map = getattr(data_config, "rename_columns", {}) or {}
        if rename_map:
            projected = [rename_map.get(c, c) for c in df.columns]
            if len(set(projected)) != len(projected):
                raise ValueError("Column rename mapping would create duplicate column names for prediction data")

            missing = [k for k in rename_map.keys() if k not in df.columns]
            if missing:
                logger.warning(f"rename_columns contains keys not found in prediction CSV: {missing}")

            df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        df = df.reset_index(drop=True)
        print(f"Successfully loaded {len(df)} rows from local path.")
        return df

    elif source == DataSource.DATABRICKS.value:
        # --- Databricks/Delta Lake Loading Logic will go here ---
        # Resolve prediction table name: allow optional `prediction_table` in config, else fall back to features_table_name
        table_name = getattr(data_config, 'prediction_table', '') or data_config.features_table_name
        table_path = f"{data_config.catalog_name}.{data_config.schema_name}.{table_name}"
        print(f"Loading prediction data from Databricks table: {table_path}")
        # Import PySpark lazily so local environments without pyspark don't fail on module import
        try:
            from pyspark.sql import SparkSession
        except Exception as e:
            raise ImportError(
                "PySpark is required to load data from Databricks but is not installed or failed to import. "
                "If you are running locally and don't need Spark, set data_source: 'local' in your config. "
                "To run Databricks integration locally install pyspark (see requirements-spark.txt) or run on Databricks. "
                f"Original error: {e}"
            )

        spark = SparkSession.builder.appName("DataLoader").getOrCreate()

        table = spark.table(table_path)

        # Apply optional renaming mapping in Spark before converting to pandas
        rename_map = getattr(data_config, "rename_columns", {}) or {}
        if rename_map:
            spark_cols = table.columns
            projected = [rename_map.get(c, c) for c in spark_cols]
            if len(set(projected)) != len(projected):
                raise ValueError("Column rename mapping would create duplicate column names (Databricks prediction)")

            for old, new in rename_map.items():
                if old in spark_cols:
                    if old != new:
                        table = table.withColumnRenamed(old, new)
                else:
                    logger.warning(f"rename_columns contains key not found in Databricks prediction table: {old}")

        df = table.toPandas().reset_index(drop=True)

        print(f"Successfully loaded {len(df)} rows from Databricks table.")
        return df
    else:
        raise ValueError(f"Invalid source '{source}'. Must be 'local' or 'databricks'.")

    
if __name__ == "__main__":
    # Example usage for local loading
    from src.config_manager import get_config

    config = get_config('config/config_base.yaml')
    df_local = load_training_data(config.data, source="local")
    print(df_local.head())

    # Example usage for Databricks loading
    df_db = load_training_data(config.data, source="databricks")
    print(df_db.head())