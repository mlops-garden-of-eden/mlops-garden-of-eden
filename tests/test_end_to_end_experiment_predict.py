import pytest
import pandas as pd
from pathlib import Path
from src.config_manager import get_config
from src.experiments_runner import ExperimentRunner
from src.predictor import ModelPredictor
import shutil

def test_end_to_end_experiment_and_predict(tmp_path):
    # Copy dataset_small.csv to a temp location
    data_path = tmp_path / "dataset_small.csv"
    orig_data = Path("data/dataset_small.csv")
    shutil.copy(orig_data, data_path)

    # Load config and patch to use the small dataset and temp model path
    config = get_config("config/config_dev.yaml")
    # Patch config to use our temp data, model path, and only one model
    config = config.__class__(**{
        **config.__dict__,
        "data": config.data.__class__(**{
            **config.data.__dict__,
            "local_train_data_path": str(data_path),
            "local_test_data_path": str(data_path),
            "features_table_name": "dataset_small.csv"
        }),
        "prediction": config.prediction.__class__(**{
            **config.prediction.__dict__,
            "model_path": str(tmp_path / "test_model.pkl"),
            "output_path": str(tmp_path / "predictions.csv")
        }),
        "artifacts": config.artifacts.__class__(**{
            **config.artifacts.__dict__,
            "save_local_models": True,
            "local_models_dir": str(tmp_path)
        }),
        "tuning": config.tuning.__class__(**{
            **config.tuning.__dict__,
            "models_to_run": ["RandomForestClassifier"]
        })
    })

    # Run experiment (should save model to tmp_path)
    runner = ExperimentRunner(config, env="dev")
    best_run_id = runner.run_experiment_pipeline()

    # Find the saved model artifact
    model_files = list(tmp_path.glob("*.pkl"))
    assert model_files, "No model artifact was saved."
    model_path = model_files[0]

    # Load model and run prediction
    predictor = ModelPredictor(config)
    predictor.load_model(str(model_path))
    df = pd.read_csv(data_path)
    # Remove target column for prediction
    X = df.drop(columns=[config.target_column]) if hasattr(config, 'target_column') else df.drop(columns=['fertilizer_name'])
    result = predictor.predict(X)
    # Should return fertilizer names, not ints
    # DEBUG: Print predictions and their types for diagnosis
    print('Predictions:', result)
    if isinstance(result, dict):
        pred = result["prediction"]
        assert isinstance(pred, str)
    else:
        assert all(isinstance(p, str) for p in result["prediction"])
