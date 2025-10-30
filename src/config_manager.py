import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Any
from pathlib import Path
from copy import deepcopy

# --- Dataclass Schemas ---

@dataclass(frozen=True)
class ModelConfig:
    """Schema for individual model definitions."""
    type: str
    hyperparameters: Dict[str, Any]

@dataclass(frozen=True)
class ModelsConfig:
    """
    Schema for the models block
    """
    RandomForestClassifier: ModelConfig
    LogisticRegression: ModelConfig
    XGBoostClassifier: ModelConfig

@dataclass(frozen=True)
class FeatureConfig:
    """Schema for defining features by type."""
    numerical: List[str]
    categorical: List[str]

@dataclass(frozen=True)
class DataConfig:
    """Schema for data source and table names"""
    catalog_name: str
    schema_name: str
    raw_table_name: str
    features_table_name: str
    local_raw_path: str
    local_train_data_path: str
    features: FeatureConfig
    rename_columns: Dict[str, str] = field(default_factory=dict)
    meta_features: List[str] = field(default_factory=lambda: ["id", "_source", "_ingestion_timestamp"])
    local_test_data_path: str = ""
    intermediate_clean_table: str = "cleaned_data" # Field with default comes last

@dataclass(frozen=True)
class FeatureOperation:
    """Schema for transformations of features"""
    output: str
    formula: str

@dataclass(frozen=True)
class FeatureEngineeringConfig:
    """Schema for feature engineering"""
    enable: bool = False
    operations: List[FeatureOperation] = field(default_factory=list)

@dataclass(frozen=True)
class TuningConfig:
    """Schema for pipeline execution settings."""
    models_to_run: List[str]
    validation_size: float

@dataclass(frozen=True)
class LogSettingsConfig:
    """Schema for logger level and file path."""
    level: str
    file_path: str

@dataclass(frozen=True)
class TrackingConfig:
    """Schema for MLflow and tracking settings."""
    experiment_name: str
    mlflow_tracking_uri: str
    log_settings: LogSettingsConfig

@dataclass(frozen=True)
class Config:
    """The main, top-level configuration schema."""
    project_name: str
    target_column: str
    random_seed: int
    data_source: str
    data: DataConfig
    tuning: TuningConfig
    tracking: TrackingConfig
    models: ModelsConfig 
    feature_engineering: FeatureEngineeringConfig = FeatureEngineeringConfig()

# --- Loading Function ---

def _load_yaml_file(file_path: Path) -> Dict[str, Any]:
    """Internal function to securely load YAML content from a given path."""
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {file_path}")
    
    with open(file_path, 'r') as f:
        # yaml.safe_load returns None for empty files, which we check for below
        return yaml.safe_load(f)

def _merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Internal function to recursively merge the override config into the base config."""
    merged_config = deepcopy(base_config)
    
    for k, v in override_config.items():
        if isinstance(v, dict) and k in merged_config and isinstance(merged_config[k], dict):
            merged_config[k] = _merge_configs(merged_config[k], v)
        elif isinstance(v, list) and k in merged_config and isinstance(merged_config[k], list):
             merged_config[k] = v
        else:
            merged_config[k] = v
            
    return merged_config

def get_config(base_path: str = 'config_base.yaml') -> Config:
    """
    Loads and validates the complete configuration using a base file.
    """
    base_config_path = Path(base_path)
    config_dict = _load_yaml_file(base_config_path)

    if config_dict is None:
        raise ValueError(f"Configuration file is empty or invalid YAML: {base_config_path}")

    # Construct Nested Dataclass Instances

    # Instantiate FeatureConfig
    data_dict = config_dict['data']
    feature_config = FeatureConfig(**data_dict.pop('features')) # Instantiate FeatureConfig

    # Instantiate DataConfig, passing the FeatureConfig instance
    data_config = DataConfig(features=feature_config, **data_dict)
    config_dict.pop('data')

    # Instantiate feature_engineering config if present in YAML
    fe_cfg = config_dict.pop("feature_engineering", None)
    if fe_cfg:
        operations_list = fe_cfg.get("operations", [])
        operations_objects = [FeatureOperation(**op) for op in operations_list]
        feature_engineering_config = FeatureEngineeringConfig(
            enable=fe_cfg.get("enable", False),
            operations=operations_objects
        )
    else:
        feature_engineering_config = FeatureEngineeringConfig()

    # Instantiate Tuning and Tracking Configs
    tuning_config = TuningConfig(**config_dict.pop('tuning'))
    
    # Instantiate Logging and Tracking Configs
    log_settings_dict = config_dict['tracking'].pop('log_settings')
    log_settings_config = LogSettingsConfig(**log_settings_dict)

    tracking_config = TrackingConfig(
        log_settings=log_settings_config,
        **config_dict.pop('tracking')
    )

    # Construct ModelsConfig 
    models_dict = config_dict.pop('models')
    models_config = ModelsConfig(
        RandomForestClassifier=ModelConfig(**models_dict.pop('RandomForestClassifier')),
        LogisticRegression=ModelConfig(**models_dict.pop('LogisticRegression')),
        XGBoostClassifier=ModelConfig(**models_dict.pop('XGBoostClassifier'))
    )

    # Construct the top-level Config
    try:
        config = Config(
        data=data_config,
        tuning=tuning_config,
        tracking=tracking_config,
        models=models_config,
        feature_engineering=feature_engineering_config,
        **config_dict
    )
    except TypeError as e:
        raise ValueError(f"Configuration validation failed for top-level Config: {e}")
        
    return config