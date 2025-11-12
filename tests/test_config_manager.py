import pytest
from pathlib import Path
from src.config_manager import (
    get_config, Config, DataConfig, ModelConfig, 
    TuningConfig, TrackingConfig, FeatureConfig 
)

@pytest.fixture
def temp_config_files(tmp_path):
    """Creates a temporary base config file for testing the base load."""
    
    # Define the complete config content matching the final dataclass structure
    base_config_content = """
    project_name: "TestProject"
    target_column: "Label"
    random_seed: 100
    data_source: "local"
    data:
      catalog_name: "base_catalog"
      schema_name: "base_schema"
      raw_table_name: "raw_data"
      features_table_name: "base_features"
      local_raw_path: "data/base_raw.csv"
      local_train_data_path: "data/base_train.csv"
      features:
        numerical: 
          - "Temparature"
          - "Humidity"
          - "Moisture"
          - "Nitrogen"
          - "Potassium"
          - "Phosphorous"
        categorical:
          - "Soil Type"
          - "Crop Type"
    tuning:
      models_to_run: ["RF", "LR", "XGB"]
      validation_size: 0.2
    tracking:
      experiment_name: "BaseExperiment"
      mlflow_tracking_uri: "local"
      log_settings:
        level: "INFO"
        file_path: "logs/test_run.log"
    models:
      RandomForestClassifier:
        type: "sklearn.RF"
        hyperparameters: {n_estimators: 10}
      LogisticRegression:
        type: "sklearn.LR"
        hyperparameters: {C: 1.0}
      XGBoostClassifier:
        type: "xgboost.XGB"
        hyperparameters: {n_estimators: 100}
    """
    
    base_file = tmp_path / "config_base.yaml"
    base_file.write_text(base_config_content)
    
    return base_file

def test_config_base_loading_simple(temp_config_files):
    """
    A simple test to verify the YAML loads and converts to the top-level 
    Config dataclass and checks one field from each major nested type.
    """
    
    # Load the config using the temporary file path
    config = get_config(str(temp_config_files))

    # Verification 1: Top-level object
    assert isinstance(config, Config), "Output is not the expected Config dataclass type."
    assert config.data_source == "local", "data_source field did not load."

    # Verification 2: Check FeatureConfig structure (data columns)
    assert isinstance(config.data.features, FeatureConfig), "Features section was not converted to FeatureConfig."
    
    # Check the actual feature lists using the provided column names
    expected_numerical = ["Temparature", "Humidity", "Moisture", "Nitrogen", "Potassium", "Phosphorous"]
    expected_categorical = ["Soil Type", "Crop Type"]
    
    assert config.data.features.numerical == expected_numerical, "Numerical features list did not load correctly."
    assert config.data.features.categorical == expected_categorical, "Categorical features list did not load correctly."
    
    # Verification 3: Check nested ModelsConfig conversion
    rf_config = config.models.RandomForestClassifier
    assert isinstance(rf_config, ModelConfig), "Individual model was not converted to ModelConfig."
    assert rf_config.type == "sklearn.RF", "Nested model type failed to load."
    assert config.tuning.validation_size == 0.2, "Validation size failed to load."