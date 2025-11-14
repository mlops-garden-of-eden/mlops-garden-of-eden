# src/experiment_runner.py

import mlflow
import pandas as pd
from datetime import datetime
from pathlib import Path
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Any, Tuple
from xgboost import XGBClassifier

from src.config_manager import Config, ModelConfig
from src.data_loader import load_training_data
from src.data_splitter import split_data
from src.model_utils import get_model_class,  get_hyperparameter_combinations
from src.preprocessor import create_preprocessor
from src.utils import logger
from itertools import combinations
from mlflow.models.signature import infer_signature


class ExperimentRunner:
    """
    Manages the entire model development lifecycle.
    """
    def __init__(self, config: Config, env: str):
        self.config = config
        self.env = env
        self.label_encoder = LabelEncoder()
        logger.info(f"Initialized Experiment Runner for environment: {self.env}")

    def load_data(self) -> pd.DataFrame:
        """Reads the clean, intermediate data by calling the dedicated loader."""

        data_source = self.config.data_source
        logger.info(f"Loading data from configured source: {data_source}")
        
        data = load_training_data(
            data_config=self.config.data,
            source=data_source
        )
        return data

    def preprocess_and_feature_engineer(self, df_train: pd.DataFrame, df_val: pd.DataFrame):
        """Stage 3: Feature engineering driven by config."""
        from src.feature_engineering import apply_feature_engineering
        fe_cfg = getattr(self.config, "feature_engineering", None)
        if not fe_cfg or not getattr(fe_cfg, "enable", False):
            logger.info("Feature engineering disabled via config.")
            return df_train, df_val, set()

        # Apply feature engineering to both train and val
        df_train_fe = apply_feature_engineering(df_train, self.config)
        df_val_fe = apply_feature_engineering(df_val, self.config)

        # Optionally, log which features were added
        added_cols = set(df_train_fe.columns) - set(df_train.columns)
        if added_cols:
            logger.info(f"Feature engineering added columns: {added_cols}")

        return df_train_fe, df_val_fe, added_cols


    def split_data(self, df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits the raw data into training and validation sets based on config.
        """
        logger.info("Splitting data into training and validation sets.")
        
        df_train, df_val = split_data(
            df=df_raw,
            target_column=self.config.target_column,
            validation_size=self.config.tuning.validation_size,
            random_seed=self.config.random_seed
        )
        logger.info(f"Train set size: {len(df_train)}, Validation set size: {len(df_val)}")
        return df_train, df_val

    def run_tuning_and_training(self, df_train: pd.DataFrame, df_val: pd.DataFrame) -> str:
        logger.info("Starting model training and hyperparameter exploration cycle.")

        best_accuracy = -1
        best_run_id = None
        best_model_artifact_name = None

        # Create a new MLFlow experiment for the entire set of models being tested
        mlflow.set_experiment(self.config.tracking.experiment_name)
        mlflow.autolog()
        iso_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Outer Loop: Iterate over all configured models
        parent_run_name = f"Experiment_{iso_timestamp}"
        with mlflow.start_run(run_name=parent_run_name) as parent_run:
            parent_run_id = parent_run.info.run_id

            for model_name in self.config.tuning.models_to_run:
                logger.info(f"--- Processing model: {model_name} ---")
                try:
                    model_config: ModelConfig = getattr(self.config.models, model_name)
                    ModelClass = get_model_class(model_config.type)
                except Exception as e:
                    logger.error(f"Failed to set up model {model_name}. Error: {e}")
                    continue

                hp_combinations = get_hyperparameter_combinations(model_config.hyperparameters)
                logger.info(f"Generated {len(hp_combinations)} hyperparameter combination(s) for {model_name}.")

                # Inner Loop: Iterate over each hyperparameter combination
                for i, hyperparams in enumerate(hp_combinations):
                    run_name = f"{model_name}_Run_{i + 1}"
                    logger.info(f"Starting run: {run_name} with HPs: {hyperparams}")

                    with mlflow.start_run(run_name=run_name, nested=True, parent_run_id=parent_run_id) as child_run:
                        # Prepare data for this run
                        X_train = df_train.drop(columns=[self.config.target_column])
                        y_train = df_train[self.config.target_column]
                        X_val = df_val.drop(columns=[self.config.target_column])
                        y_val = df_val[self.config.target_column]

                        # Fit label encoder on y_train
                        label_encoder = LabelEncoder()
                        y_train_enc = label_encoder.fit_transform(y_train)
                        y_val_enc = label_encoder.transform(y_val)

                        # Instantiate the model with the specific combination of hyperparameters
                        classifier = ModelClass(random_state=self.config.random_seed, **hyperparams)
                        # Create the reusable preprocessor
                        numerical_features = self.config.data.features.numerical
                        categorical_features = self.config.data.features.categorical
                        meta_features = getattr(self.config.data, 'meta_features', [])
                        preprocessor = create_preprocessor(numerical_features, categorical_features, meta_features=meta_features)

                        full_pipeline = Pipeline(steps=[
                            ('preprocessor', preprocessor),
                            ('classifier', classifier)
                        ])

                        # Training
                        full_pipeline.fit(X_train, y_train_enc)

                        # Compute Training Accuracy
                        y_pred_train = full_pipeline.predict(X_train)
                        train_acc = accuracy_score(y_train_enc, y_pred_train)

                        # Compute Validation Accuracy
                        y_pred_val = full_pipeline.predict(X_val)
                        val_acc = accuracy_score(y_val_enc, y_pred_val)
                        mlflow.log_metric("train_accuracy", train_acc)
                        mlflow.log_metric("val_accuracy", val_acc)

                        train_preds = full_pipeline.predict(X_train)
                        signature = infer_signature(X_train, train_preds)

                        # Save both pipeline and label_encoder as a tuple
                        pipeline_and_encoder = (full_pipeline, label_encoder)

                        # Log the model with signature and input example
                        mlflow.sklearn.log_model(
                            sk_model=full_pipeline,
                            artifact_path=model_name,
                            signature=signature,
                            input_example=X_train.head(5)  # optional example
                        )
                        
                        # Log label encoder as a separate artifact
                        import tempfile
                        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
                            pickle.dump(label_encoder, f)
                            temp_path = f.name
                        try:
                            mlflow.log_artifact(temp_path, artifact_path=f"{model_name}_label_encoder")
                            logger.info(f"Logged label encoder as artifact: {model_name}_label_encoder")
                        finally:
                            Path(temp_path).unlink()  # Clean up temp file

                        # Optionally save a local copy of the trained pipeline for quick local inference
                        artifacts_cfg = getattr(self.config, 'artifacts', None)
                        try:
                            save_local_flag = getattr(artifacts_cfg, 'save_local_models', False) if artifacts_cfg is not None else False
                        except Exception:
                            save_local_flag = False

                        if save_local_flag:
                            local_dir = Path(getattr(artifacts_cfg, 'local_models_dir', 'models'))
                            local_dir.mkdir(parents=True, exist_ok=True)
                            local_path = local_dir / f"{model_name}_{child_run.info.run_id}.pkl"
                            try:
                                with open(local_path, 'wb') as _f:
                                    pickle.dump(pipeline_and_encoder, _f)
                                logger.info(f"Saved local model artifact to {local_path}")
                            except Exception as e:
                                logger.warning(f"Failed to save local model artifact to {local_path}: {e}")

                        if val_acc > best_accuracy:
                            best_accuracy = val_acc
                            best_run_id = child_run.info.run_id
                            best_model_artifact_name = model_name

        # Register best model
        if best_run_id and best_model_artifact_name:
            model_uri = f"runs:/{best_run_id}/{best_model_artifact_name}"
            registered_model = mlflow.register_model(model_uri, "BestFertilizerModel")

            client = mlflow.tracking.MlflowClient()
            client.set_tag(best_run_id, "best_val_accuracy", str(best_accuracy))
            client.set_tag(best_run_id, "timestamp", iso_timestamp)

        logger.info(f"--- All experiments finished. Best run: {best_run_id} (Accuracy: {best_accuracy:.4f}) ---")
        return best_run_id


    def run_experiment_pipeline(self) -> str:
        """
        Orchestrates the entire experiment process: Load -> Split -> Train.
        
        Returns:
            The unique identifier (run_id) of the best model's experiment.
        """
        logger.info("Starting MLOps experiment pipeline orchestration.")
        try:
            # Load Data
            df_raw = self.load_data()

            # Split Data
            df_train, df_val = self.split_data(df_raw)

            # Preprocess and Train (The Core Experimentation)
            # Calls the method that loops through models, preprocesses, trains, and evaluates
            df_train, df_val, applied_features = self.preprocess_and_feature_engineer(df_train, df_val)
            logger.info(f"Applied features (pipeline): {applied_features}") # <-- can replace with MLflow integration
            best_run_id = self.run_tuning_and_training(df_train, df_val)

            logger.info("Pipeline execution completed successfully.")
            return best_run_id

        except NotImplementedError as e:
            logger.warning(f"Pipeline skipped critical stage: {e}")
            logger.warning("Execution terminated gracefully due to unimplemented feature.")
            raise

        except Exception as e:
            logger.critical(f"Pipeline terminated with a CRITICAL failure: {e}", exc_info=True)
            raise

def run_experiment_stage(config: Config, env: str) -> str:
    """
    External entry point called by the main pipeline script.
    """
    runner = ExperimentRunner(config, env)
    return runner.run_experiment_pipeline()