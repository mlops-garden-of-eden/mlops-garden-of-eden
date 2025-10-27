# src/experiment_runner.py

import pandas as pd
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
        fe_cfg = getattr(self.config, "feature_engineering", None)
        if not fe_cfg or not getattr(fe_cfg, "enable", False):
            logger.info("Feature engineering disabled via config.")
            return df_train, df_val

        applied_features = []

        operations = getattr(fe_cfg, "operations", []) or []
        if not operations:
            logger.info("No feature engineering operations defined in config.")
            return df_train, df_val

        for op in operations:
            output_col = op.output
            formula = op.formula
            if not output_col or not formula:
                logger.warning(f"Skipping invalid feature operation (missing output/formula): {op}")
                continue

            try:
                df_train[output_col] = df_train.eval(formula)
                df_val[output_col] = df_val.eval(formula)
                applied_features.append(output_col)  # <-- track the feature
                logger.info(f"Created numeric feature '{output_col}' using formula: {formula}")
            except Exception as e:
                logger.warning(f"Failed to create numeric feature '{output_col}': {e}")

        logger.info(f"Numeric features successfully applied this run: {applied_features}") #<-- print

        return df_train, df_val, applied_features

    def _transform_labels(self, df_train: pd.DataFrame, df_val: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Encodes categorical labels (y) into numerical integers [0, 1, 2, ...].
        """
        target_col = self.config.target_column

        # FIT and TRANSFORM on Training Data
        logger.info(f"Fitting LabelEncoder on training labels ({target_col}).")
        y_train_encoded = self.label_encoder.fit_transform(df_train[target_col])
        
        # TRANSFORM only on Validation Data (using the fitted encoder)
        y_val_encoded = self.label_encoder.transform(df_val[target_col])

        # Replace the original target columns with the encoded values
        df_train[target_col] = y_train_encoded
        df_val[target_col] = y_val_encoded
        
        logger.info(f"Labels transformed to numerical: {self.label_encoder.classes_}")
        return df_train, df_val

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
        
        X_train = df_train.drop(columns=[self.config.target_column])
        y_train = df_train[self.config.target_column]
        X_val = df_val.drop(columns=[self.config.target_column])
        y_val = df_val[self.config.target_column]

        results = {}
        best_accuracy = -1
        best_run_name = ""
        
        # Outer Loop: Iterate over all configured models 
        for model_name in self.config.tuning.models_to_run:
            logger.info(f"--- Processing model: {model_name} ---")
            
            try:
                model_config: ModelConfig = getattr(self.config.models, model_name)
                ModelClass = get_model_class(model_config.type)
            except Exception as e:
                logger.error(f"Failed to set up model {model_name}. Error: {e}")
                continue

            # Generate all hyperparameter combinations (Grid Search)
            hp_combinations = get_hyperparameter_combinations(model_config.hyperparameters)
            logger.info(f"Generated {len(hp_combinations)} hyperparameter combination(s) for {model_name}.")

            # Inner Loop: Iterate over each hyperparameter combination
            for i, hyperparams in enumerate(hp_combinations):
                run_name = f"{model_name}_Run_{i + 1}"
                logger.info(f"Starting run: {run_name} with HPs: {hyperparams}")
                
                # Build Full Pipeline and Instantiate Model
                
                # Instantiate the model with the specific combination of hyperparameters
                classifier = ModelClass(random_state=self.config.random_seed, **hyperparams)
                
                # Create the reusable preprocessor
                numerical_features = self.config.data.features.numerical
                categorical_features = self.config.data.features.categorical
                preprocessor = create_preprocessor(numerical_features, categorical_features)

                full_pipeline = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('classifier', classifier)
                ])

                # Training
                full_pipeline.fit(X_train, y_train)

                # Compute Training Accuracy
                y_pred_train = full_pipeline.predict(X_train)
                train_acc = accuracy_score(y_train, y_pred_train)

                # Compute Validation Accuracy
                y_pred_val = full_pipeline.predict(X_val)
                val_acc = accuracy_score(y_val, y_pred_val)

                logger.info(f"Train Accuracy: {train_acc:.4f} | Validation Accuracy: {val_acc:.4f}")

                # Store Results and Track Best Model
                # NOTE: In MLFlow you will log model_name, hyperparams, and metrics for this run

                results[run_name] = {"train_accuracy": train_acc, "val_accuracy": val_acc, "model": full_pipeline}
                if val_acc > best_accuracy:
                    best_accuracy = val_acc
                    best_run_name = run_name

        logger.info(f"--- All experiments finished. Best run: {best_run_name} (Accuracy: {best_accuracy:.4f}) ---")
        
        return f"Placeholder_RunID_{best_run_name}"

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

            # Transform Labels
            df_train, df_val = self._transform_labels(df_train, df_val)
            
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