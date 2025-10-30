import pandas as pd
from pathlib import Path
from src.data_loader import load_training_data
from src.config_manager import DataConfig, FeatureConfig


def test_local_csv_rename_columns(tmp_path):
    # Create a small CSV with old-style column names
    df = pd.DataFrame({
        "Temparature": [10, 20],
        "Soil Type": ["A", "B"],
        "Label": [0, 1]
    })

    csv_path = tmp_path / "sample.csv"
    df.to_csv(csv_path, index=False)

    # Build a minimal DataConfig pointing to this file
    data_cfg = DataConfig(
        catalog_name="cat",
        schema_name="sch",
        raw_table_name="raw",
        features_table_name="features",
        local_raw_path="",
        local_train_data_path=str(csv_path),
        features=FeatureConfig(numerical=["Temparature"], categorical=["Soil Type"]),
        rename_columns={"Temparature": "temperature", "Soil Type": "soil_type"}
    )

    loaded = load_training_data(data_cfg, source="local")

    # Assert renamed columns exist and old names do not
    assert "temperature" in loaded.columns
    assert "soil_type" in loaded.columns
    assert "Temparature" not in loaded.columns
    assert "Soil Type" not in loaded.columns
