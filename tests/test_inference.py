import pytest
import pandas as pd
import pickle
from pathlib import Path
from src.config_manager import get_config
from src.predictor import run_prediction, ModelPredictor

@pytest.fixture
def dummy_pipeline(tmp_path):
    # Create a dummy sklearn pipeline and save it
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    import numpy as np
    X = np.array([[1,2],[3,4],[5,6]])
    y = np.array([0,1,0])
    # Fit on DataFrame with feature names to avoid warnings
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression())
    ])
    pipe.fit(df, y)
    model_path = tmp_path / "dummy_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(pipe, f)
    return model_path, X, y

def test_run_prediction_with_dataframe(tmp_path, dummy_pipeline):
    model_path, X, y = dummy_pipeline
    # Minimal config with required fields
    config = get_config()
    # Patch config to use our dummy model
    config = config.__class__(**{**config.__dict__, "prediction": config.prediction.__class__(**{**config.prediction.__dict__, "model_path": str(model_path)})})
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    # Should not raise
    result = run_prediction(config, df, model_path=str(model_path))
    assert hasattr(result, "shape")
    assert len(result) == len(df)

def test_model_predictor_single_sample(tmp_path, dummy_pipeline):
    model_path, X, y = dummy_pipeline
    config = get_config()
    config = config.__class__(**{**config.__dict__, "prediction": config.prediction.__class__(**{**config.prediction.__dict__, "model_path": str(model_path)})})
    predictor = ModelPredictor(config)
    predictor.load_model(str(model_path))
    sample = {f"f{i}": X[0, i] for i in range(X.shape[1])}
    result = predictor.predict(sample)
    assert isinstance(result, dict)
    assert "prediction" in result
    assert "probability_class_0" in result or "probability" in result

def test_model_predictor_batch(tmp_path, dummy_pipeline):
    model_path, X, y = dummy_pipeline
    config = get_config()
    config = config.__class__(**{**config.__dict__, "prediction": config.prediction.__class__(**{**config.prediction.__dict__, "model_path": str(model_path)})})
    predictor = ModelPredictor(config)
    predictor.load_model(str(model_path))
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    result = predictor.predict_batch(df, batch_size=2)
    assert hasattr(result, "shape")
    assert len(result) == len(df)
