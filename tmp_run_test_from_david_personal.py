import sys
from pathlib import Path
import mlflow
from mlflow import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from mlflow.models.signature import infer_signature

# ----------------------------
# Paths & repository root
# ----------------------------
try:
    repo_root = Path(__file__).resolve().parent
except NameError:
    # __file__ is not defined in Databricks job execution
    repo_root = Path.cwd().resolve()

data_path = repo_root / "data" / "dataset_small.csv"

# ----------------------------
# MLflow experiment setup
# ----------------------------
experiment_name = "/Users/daveed@cs.toronto.edu/mlops-garden-of-eden-personal/test_runs"
mlflow.set_experiment(experiment_name)

# ----------------------------
# Load data
# ----------------------------
from src.data_loader import load_training_data
from src.config_manager import DataConfig, FeatureConfig

data_cfg = DataConfig(
    catalog_name="cat",
    schema_name="sch",
    raw_table_name="raw",
    features_table_name="features",
    local_raw_path="",
    local_train_data_path=str(data_path),
    features=FeatureConfig(
        numerical=["nitrogen", "phosphorous", "potassium", "moisture", "temperature", "humidity"],
        categorical=["soil_type", "fertilizer_name", "crop_type"]
    ),
    rename_columns={}
)

try:
    df = load_training_data(data_cfg, source="local")
    print(f"Loaded {len(df)} rows from {data_path}")
except FileNotFoundError:
    raise FileNotFoundError(f"Data file not found at {data_path}. Ensure your copy includes it.")

# ----------------------------
# Preprocessing
# ----------------------------
X = df.drop(columns=["fertilizer_name", "soil_type", "crop_type"])
y = df["fertilizer_name"]

if {"nitrogen", "phosphorous", "potassium"}.issubset(df.columns):
    X["npk_ratio"] = df["nitrogen"] / (df["phosphorous"] + df["potassium"] + 1e-5)

if {"moisture", "temperature", "humidity"}.issubset(df.columns):
    X["moisture_temp_humidity_interaction"] = df["moisture"] * df["temperature"] * df["humidity"]

# --- Cast all numeric features to float to avoid MLflow integer schema warnings ---
numeric_cols = X.select_dtypes(include=["int", "int64"]).columns
X[numeric_cols] = X[numeric_cols].astype(float)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# ----------------------------
# Model training & logging
# ----------------------------
models = {
    "RandomForestClassifier": RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=5, class_weight="balanced"
    ),
    "LogisticRegression": LogisticRegression(
        penalty="l2", C=1.0, solver="saga", max_iter=500, class_weight="balanced"
    ),
    "XGBoostClassifier": XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1, objective="multi:softprob"
    ),
}

for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        train_acc = model.score(X_train, y_train)
        val_acc = model.score(X_val, y_val)

        # Infer signature and use a sample for input_example
        input_example = X_train.head(5)
        signature = infer_signature(X_train, model.predict(X_train))

        # Log metrics & model
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("val_accuracy", val_acc)
        mlflow.sklearn.log_model(
            sk_model=model,
            name=model_name,
            signature=signature,
            input_example=input_example
        )

        print(f"{model_name}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

print("All models trained and logged to MLflow successfully.")