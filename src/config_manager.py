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
class PredictionConfig:
    """Schema for prediction/inference settings."""
    model_path: str
    return_probabilities: bool = True
    output_path: str = "predictions/prediction_results.csv"


@dataclass(frozen=True)
class ArtifactsConfig:
    """Schema for artifact storage settings used by experiments and runs."""
    save_local_models: bool = False
    local_models_dir: str = "models"

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
    prediction: PredictionConfig = PredictionConfig(model_path="models/latest_model.pkl")
    artifacts: ArtifactsConfig = ArtifactsConfig()
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

def get_config(config_path: str = 'config/config_base.yaml') -> Config:
    """
    Load configuration by merging the base config and an optional environment-specific override.

    Behavior:
    - If `config_path` points to `config_base.yaml`, it is loaded directly.
    - If `config_path` points to an environment file (e.g. `config_dev.yaml` or `config_prod.yaml`),
      the function will look for `config_base.yaml` in the same directory and merge base <- override
      so that the override only needs to specify differing keys.
    """
    config_path = Path(config_path)

    # If the caller passed an env override (not the base file), try to merge it with base
    if config_path.name != 'config_base.yaml':
        override_path = config_path
        base_path = config_path.parent / 'config_base.yaml'
        if base_path.exists():
            base_cfg = _load_yaml_file(base_path) or {}
            override_cfg = _load_yaml_file(override_path) or {}
            config_dict = _merge_configs(base_cfg, override_cfg)
        else:
            # No base found, load the provided file alone
            config_dict = _load_yaml_file(config_path) or {}
    else:
        # Directly load base config
        config_dict = _load_yaml_file(config_path) or {}

    if not config_dict:
        raise ValueError(f"Configuration file is empty or invalid YAML: {config_path}")

    # Construct Nested Dataclass Instances

    # Instantiate FeatureConfig and DataConfig
    data_dict = config_dict.get('data', {})
    feature_config = FeatureConfig(**data_dict.pop('features')) if data_dict.get('features') else FeatureConfig(numerical=[], categorical=[])
    data_config = DataConfig(features=feature_config, **data_dict) if data_dict else DataConfig(features=feature_config)
    config_dict.pop('data', None)

    # Instantiate PredictionConfig (optional)
    pred_cfg = config_dict.pop('prediction', None)
    if pred_cfg:
        prediction_config = PredictionConfig(**pred_cfg)
    else:
        # default prediction config
        prediction_config = PredictionConfig(model_path="models/latest_model.pkl", return_probabilities=True, output_path="predictions/prediction_results.csv")

    # Instantiate ArtifactsConfig (optional)
    artifacts_cfg = config_dict.pop('artifacts', None)
    if artifacts_cfg:
        artifacts_config = ArtifactsConfig(**artifacts_cfg)
    else:
        artifacts_config = ArtifactsConfig()

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
            prediction=prediction_config,
            artifacts=artifacts_config,
            feature_engineering=feature_engineering_config,
            **config_dict
        )
    except TypeError as e:
        raise ValueError(f"Configuration validation failed for top-level Config: {e}")

    return config