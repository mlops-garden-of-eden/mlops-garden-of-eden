import pytest
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from src.predictor import ModelPredictor
from src.config_manager import get_config


class DummyLabelEncoder:
    def __init__(self, classes):
        self._classes = np.array(classes)
    def fit(self, y):
        return self
    def transform(self, y):
        return np.array([np.where(self._classes == v)[0][0] for v in y])
    def inverse_transform(self, y):
        return self._classes[y]
    def classes_(self):
        return self._classes

class DummyPipeline:
    def __init__(self, y_true=None, y_pred=None):
        self._y_true = y_true
        self._y_pred = y_pred
    def predict(self, X):
        if self._y_pred is not None:
            return self._y_pred
        return np.zeros(len(X), dtype=int)
    def predict_proba(self, X):
        n_classes = len(np.unique(self._y_true)) if self._y_true is not None else 1
        return np.ones((len(X), n_classes)) / n_classes

def make_artifact(tmp_path, y_true, y_pred=None):
    encoder = DummyLabelEncoder(classes=np.unique(y_true))
    artifact = (DummyPipeline(y_true=y_true, y_pred=y_pred), encoder)
    path = tmp_path / "artifact.pkl"
    with open(path, "wb") as f:
        pickle.dump(artifact, f)
    return path, encoder

def test_label_encoder_mapping(tmp_path):
    y_true = np.array(["A", "B", "C", "A"])
    X = pd.DataFrame({"f1": [1,2,3,4]})
    path, encoder = make_artifact(tmp_path, y_true, y_pred=np.array([0,1,2,0]))
    config = get_config()
    config = config.__class__(**{**config.__dict__, "prediction": config.prediction.__class__(**{**config.prediction.__dict__, "model_path": str(path)})})
    predictor = ModelPredictor(config)
    predictor.load_model(str(path))
    result = predictor.predict(X)
    # Should map to fertilizer names (not ints)
    if isinstance(result, dict):
        pred = result["prediction"]
        assert pred in encoder._classes
    else:
        assert all(p in encoder._classes for p in result["prediction"])

def test_label_encoder_missing(tmp_path):
    # Simulate artifact without label encoder
    artifact = DummyPipeline()
    path = tmp_path / "artifact_no_encoder.pkl"
    with open(path, "wb") as f:
        pickle.dump(artifact, f)
    config = get_config()
    config = config.__class__(**{**config.__dict__, "prediction": config.prediction.__class__(**{**config.prediction.__dict__, "model_path": str(path)})})
    predictor = ModelPredictor(config)
    predictor.load_model(str(path))
    X = pd.DataFrame({"f1": [1,2,3]})
    result = predictor.predict(X)
    # Should return ints, not mapped
    if isinstance(result, dict):
        pred = result["prediction"]
        assert isinstance(pred, (int, np.integer))
    else:
        assert all(isinstance(p, (int, np.integer)) for p in result["prediction"])
