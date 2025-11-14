"""
Pytest tests for MLflow model loading from Databricks.
Run with: pytest test_mlflow_model_loading.py -v
"""

import os
import pytest
import mlflow
from src.config_manager import get_config
from src.predictor import ModelPredictor


@pytest.fixture(scope="module")
def databricks_credentials():
    """Verify Databricks credentials are set."""
    host = os.environ.get("DATABRICKS_HOST")
    token = os.environ.get("DATABRICKS_TOKEN")
    
    if not host or not token:
        pytest.skip("DATABRICKS_HOST and DATABRICKS_TOKEN must be set")
    
    return {"host": host, "token": token}


@pytest.fixture(scope="module")
def prod_config():
    """Load production configuration."""
    return get_config("config/config_prod.yaml")


@pytest.fixture(scope="module")
def predictor(prod_config, databricks_credentials):
    """Create and load predictor with model."""
    mlflow.set_tracking_uri("databricks")
    predictor = ModelPredictor(prod_config, mode="production")
    predictor.load_model()
    return predictor


def test_mlflow_connection(databricks_credentials):
    """Test MLflow connection to Databricks."""
    mlflow.set_tracking_uri("databricks")
    assert mlflow.get_tracking_uri() == "databricks"


def test_experiment_exists(prod_config, databricks_credentials):
    """Test that the configured experiment exists."""
    mlflow.set_tracking_uri("databricks")
    experiment_name = prod_config.tracking.experiment_name
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    assert experiment is not None, f"Experiment '{experiment_name}' not found"
    
    # Check if there are runs
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=1
    )
    assert not runs.empty, "No runs found in experiment"


def test_model_loads(predictor):
    """Test that model loads successfully."""
    assert predictor.model is not None, "Model not loaded"
    assert hasattr(predictor.model, 'predict'), "Model doesn't have predict method"


def test_label_encoder_loaded(predictor):
    """Test that label encoder is loaded."""
    assert predictor.label_encoder is not None, (
        "Label encoder not loaded. Model may be from before the fix. "
        "Retrain with: databricks bundle run RunExperiments -t dev"
    )
    assert hasattr(predictor.label_encoder, 'classes_'), "Invalid label encoder"
    assert len(predictor.label_encoder.classes_) > 0, "Label encoder has no classes"


def test_prediction(predictor):
    """Test making a sample prediction."""
    sample_input = {
        "temperature": 25.0,
        "humidity": 60.0,
        "moisture": 45.0,
        "nitrogen": 40.0,
        "potassium": 30.0,
        "phosphorous": 35.0,
        "soil_type": "Sandy",
        "crop_type": "Wheat"
    }
    
    result = predictor.predict(sample_input, source="local", return_input=False)
    
    assert result is not None, "Prediction returned None"
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert "prediction" in result, "No 'prediction' in result"
    
    # Check if prediction is a string (fertilizer name) not numeric
    prediction = result["prediction"]
    assert isinstance(prediction, str), (
        f"Prediction is numeric ({prediction}). Label encoder may not be working."
    )
