"""
Prediction pipeline for running inference on trained models.
"""
import pandas as pd
import pickle
from pathlib import Path
from typing import Union, Optional, Literal, Dict, Any
import json
from datetime import datetime

from .config_manager import Config
from .utils import logger
from .data_loader import load_prediction_data


class ModelPredictor:
    """
    Handles loading trained models and running predictions.
    Supports both dev and production modes.
    """
    
    def __init__(
        self,
        config: Config,
        mode: Literal["dev", "production"] = "dev"
    ):
        """
        Initialize predictor with configuration.
        
        Args:
            config: Full configuration object
            mode: Execution mode ('dev' or 'production')
        """
        self.config = config
        self.mode = mode
        self.model = None
        self.preprocessor = None
        self.metadata = {}
        
        # Mode-specific settings
        self.verbose = mode == "dev"
        # Read return_probabilities from config.prediction when available, else default to dev behavior
        pred_cfg = getattr(config, 'prediction', None)
        if pred_cfg is not None:
            self.return_probabilities = getattr(pred_cfg, 'return_probabilities', mode == "dev")
        else:
            self.return_probabilities = mode == "dev"
        
    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        Load a trained pipeline (model + preprocessor as a single object).
        Supports both local file paths and MLflow URIs.
        
        Args:
            model_path: Path to saved model. Can be:
                - Local file path: "models/model.pkl"
                - MLflow run URI: "runs:/{run_id}/{artifact_path}"
                - MLflow models URI: "models:/{model_name}/{version_or_stage}"
        """
        if model_path is None:
            pred_cfg = getattr(self.config, 'prediction', None)
            model_path = getattr(pred_cfg, 'model_path', None) if pred_cfg is not None else None
            if model_path is None:
                logger.error("Model path not provided to load_model() and no 'prediction.model_path' found in config.")
                raise ValueError("Model path not provided to load_model() and no 'prediction.model_path' found in config.")

        # Check if it's an MLflow URI
        if isinstance(model_path, str) and (model_path.startswith("runs:/") or model_path.startswith("models:/")):
            logger.info(f"Loading model from MLflow URI: {model_path}")
            try:
                import mlflow
                import tempfile
                
                # Set MLflow tracking URI from config if available
                tracking_cfg = getattr(self.config, 'tracking', None)
                if tracking_cfg is not None:
                    mlflow_uri = getattr(tracking_cfg, 'mlflow_tracking_uri', None)
                    if mlflow_uri:
                        mlflow.set_tracking_uri(mlflow_uri)
                        logger.info(f"Set MLflow tracking URI to: {mlflow_uri}")
                
                # Load model from MLflow
                loaded = mlflow.sklearn.load_model(model_path)
                
                # MLflow models are typically just the pipeline, not (pipeline, encoder) tuple
                if isinstance(loaded, tuple) and len(loaded) == 2:
                    self.model, self.label_encoder = loaded
                    logger.info("Loaded pipeline and label encoder from MLflow artifact.")
                else:
                    self.model = loaded
                    logger.info("Loaded model from MLflow. Attempting to load label encoder separately...")
                    
                    # Try to load label encoder from separate artifact
                    # Extract run_id and artifact_path from model_path
                    try:
                        if model_path.startswith("runs:/"):
                            # Format: runs:/{run_id}/{artifact_path}
                            parts = model_path.replace("runs:/", "").split("/", 1)
                            run_id = parts[0]
                            artifact_path = parts[1] if len(parts) > 1 else ""
                            
                            # Download label encoder artifact
                            encoder_artifact_path = f"{artifact_path}_label_encoder"
                            logger.info(f"Looking for label encoder at: {encoder_artifact_path}")
                            
                            client = mlflow.tracking.MlflowClient()
                            temp_dir = tempfile.mkdtemp()
                            try:
                                # download_artifacts returns the local path to the downloaded artifact
                                artifact_uri = client.download_artifacts(run_id, encoder_artifact_path, temp_dir)
                                artifact_path_obj = Path(artifact_uri)
                                
                                logger.info(f"Downloaded artifact to: {artifact_path_obj}")
                                
                                # The downloaded artifact could be either:
                                # 1. A directory containing the pickle file
                                # 2. The pickle file itself
                                
                                possible_files = []
                                
                                if artifact_path_obj.is_file():
                                    # It's the file itself
                                    possible_files.append(artifact_path_obj)
                                elif artifact_path_obj.is_dir():
                                    # It's a directory, look for .pkl files
                                    pkl_files = list(artifact_path_obj.glob("*.pkl"))
                                    possible_files.extend(pkl_files)
                                    # Also try specific names
                                    possible_files.append(artifact_path_obj / "label_encoder.pkl")
                                    possible_files.append(artifact_path_obj / f"{artifact_path}.pkl")
                                
                                logger.info(f"Trying to load from: {[str(f) for f in possible_files]}")
                                
                                encoder_loaded = False
                                for enc_path in possible_files:
                                    if enc_path.exists() and enc_path.is_file():
                                        logger.info(f"Found encoder file: {enc_path}")
                                        try:
                                            with open(enc_path, 'rb') as f:
                                                self.label_encoder = pickle.load(f)
                                            logger.info(f"✅ Loaded label encoder from: {enc_path}")
                                            encoder_loaded = True
                                            break
                                        except Exception as e:
                                            logger.warning(f"Failed to load from {enc_path}: {e}")
                                            continue
                                
                                if not encoder_loaded:
                                    logger.warning(f"Label encoder artifact not found. Predictions will be numeric.")
                                    logger.warning(f"Checked path: {artifact_path_obj}")
                                    self.label_encoder = None
                                    
                            except Exception as e:
                                logger.warning(f"Could not load label encoder: {e}. Predictions will be numeric.")
                                self.label_encoder = None
                            finally:
                                # Cleanup temp directory
                                import shutil
                                try:
                                    shutil.rmtree(temp_dir)
                                except:
                                    pass
                        else:
                            # For models:/ URIs, we can't easily get the label encoder
                            logger.warning("Label encoder not available for model registry URIs. Predictions will be numeric.")
                            self.label_encoder = None
                            
                    except Exception as e:
                        logger.warning(f"Error loading label encoder: {e}. Predictions will be numeric.")
                        self.label_encoder = None
                
                self.preprocessor = None
                self.metadata = {}
                logger.info(f"Pipeline loaded successfully from MLflow. Type: {type(self.model).__name__}")
                
            except Exception as e:
                logger.error(f"Failed to load model from MLflow URI: {e}")
                raise
        else:
            # Load from local file path
            model_path = Path(model_path)
            if not model_path.exists():
                logger.error(f"Model not found at: {model_path}")
                raise FileNotFoundError(f"Model not found at: {model_path}")

            logger.info(f"Loading model pipeline from local file: {model_path}")

            # Load (pipeline, label_encoder) tuple
            with open(model_path, 'rb') as _f:
                loaded = pickle.load(_f)
            if isinstance(loaded, tuple) and len(loaded) == 2:
                self.model, self.label_encoder = loaded
                logger.info("Loaded pipeline and label encoder from artifact.")
            else:
                self.model = loaded
                self.label_encoder = None
                logger.warning("Loaded artifact is not a (pipeline, label_encoder) tuple. Predictions will not be mapped to fertilizer names.")
            self.preprocessor = None  # Not needed; pipeline handles preprocessing
            self.metadata = {}
            logger.info(f"Pipeline loaded successfully. Type: {type(self.model).__name__}")
    
    def validate_input(self, df: pd.DataFrame) -> None:
        # No-op: pipeline handles feature selection internally
        pass
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        # No-op: pipeline handles preprocessing
        return df
    
    def predict(
        self,
        input_data: Union[pd.DataFrame, dict, str],
        source: Literal["local", "databricks"] = "local",
        return_input: bool = False
    ) -> Union[pd.DataFrame, Dict[str, Any]]:
        """
        Run prediction on input data.
        
        Args:
            input_data: Input data (DataFrame, dict for single sample, or file path)
            source: Data source type
            return_input: Whether to include input features in output
            
        Returns:
            Predictions as DataFrame or dict (for single sample)
        """
        if self.model is None:
            logger.error("Model not loaded. Call load_model() first.")
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Load and prepare data
        # Support passing a DataFrame, a dict (single sample), a local CSV path, or None to use config paths

        if isinstance(input_data, pd.DataFrame):
            logger.info(f"Received input as DataFrame with shape {input_data.shape}")
            df = input_data.copy()
        elif isinstance(input_data, dict):
            logger.info("Received input as dict (single sample)")
            # Remove 'id' if present in dict
            input_data = {k: v for k, v in input_data.items() if k != 'id'}
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, str):
            p = Path(input_data)
            if p.exists():
                logger.info(f"Loading input data from file: {input_data}")
                df = pd.read_csv(p)
            else:
                logger.info(f"Loading prediction data using configured loader: {input_data}")
                df = load_prediction_data(data_config=self.config.data, source=source)
        elif input_data is None:
            logger.info("No input_data provided, loading prediction data from config.")
            df = load_prediction_data(data_config=self.config.data, source=source)
        else:
            logger.error("Unsupported input_data type for prediction. Pass DataFrame, dict, path str, or None.")
            raise ValueError("Unsupported input_data type for prediction. Pass DataFrame, dict, path str, or None.")

        # Always drop 'id' column if present
        if 'id' in df.columns:
            df = df.drop(columns=['id'])

        # --- Feature engineering: apply same logic as training ---
        try:
            from src.feature_engineering import apply_feature_engineering
            df = apply_feature_engineering(df, self.config)
            logger.info("Applied feature engineering to input data.")
        except ImportError:
            logger.warning("Feature engineering module not found; skipping feature engineering.")
            pass

        original_df = df.copy()
        is_single_sample = len(df) == 1 and isinstance(input_data, dict)

        logger.info(f"Running predictions on {len(df)} samples...")


        predictions = self.model.predict(df)
        # If label_encoder is present, map predictions to fertilizer names
        logger.info('Label encoder: %s', self.label_encoder)
        logger.info('Label encoder classes: %s', getattr(self.label_encoder, 'classes_', None))
        logger.info('Predictions before mapping: %s', predictions[:10])
        if self.label_encoder is not None:
            try:
                mapped = self.label_encoder.inverse_transform(predictions)
                logger.info('Predictions after mapping: %s', mapped[:10])
                predictions = mapped
            except Exception as e:
                print('Mapping error:', e)
                logger.warning(f"Could not map predictions to fertilizer names: {e}")

        # Get probabilities if available and requested
        probabilities = None
        if self.return_probabilities and hasattr(self.model, 'predict_proba'):
            logger.info("Model supports predict_proba; returning probabilities as well.")
            probabilities = self.model.predict_proba(df)

        # Format results
        results = self._format_results(
            predictions=predictions,
            probabilities=probabilities,
            original_data=original_df if return_input else None,
            is_single_sample=is_single_sample
        )

        logger.info("Predictions completed successfully.")

        return results
    
    def _format_results(
        self,
        predictions: Any,
        probabilities: Optional[Any],
        original_data: Optional[pd.DataFrame],
        is_single_sample: bool
    ) -> Union[pd.DataFrame, Dict[str, Any]]:
        """
        Format prediction results based on mode and input type.
        
        Args:
            predictions: Model predictions
            probabilities: Prediction probabilities (if available)
            original_data: Original input data
            is_single_sample: Whether input was a single sample
            
        Returns:
            Formatted results
        """
        # Create results dataframe
        results_df = pd.DataFrame({'prediction': predictions})
        
        # Add probabilities if available
        if probabilities is not None:
            if probabilities.ndim == 1:
                results_df['probability'] = probabilities
            else:
                # Multi-class: add probability for each class
                for i in range(probabilities.shape[1]):
                    results_df[f'probability_class_{i}'] = probabilities[:, i]
        
        # Add timestamp in production mode
        if self.mode == "production":
            results_df['prediction_timestamp'] = datetime.now().isoformat()
        
        # Include original input if requested
        if original_data is not None:
            results_df = pd.concat([original_data.reset_index(drop=True), results_df], axis=1)
        
        # Return as dict for single sample (more convenient for APIs)
        if is_single_sample:
            result_dict = results_df.iloc[0].to_dict()
            
            # Add metadata in dev mode
            if self.mode == "dev" and self.metadata:
                result_dict['_metadata'] = {
                    'model_type': self.metadata.get('model_type'),
                    'trained_on': self.metadata.get('training_date')
                }
            
            return result_dict
        
        return results_df
    
    def predict_batch(
        self,
        df: pd.DataFrame,
        batch_size: int = 1000,
        source: Literal["local", "databricks"] = "local"
    ) -> pd.DataFrame:
        """
        Run predictions on large datasets in batches.
        Useful for production mode with large data volumes.
        
        Args:
            df: Input dataframe
            batch_size: Number of samples per batch
            source: Data source type
            
        Returns:
            Complete predictions dataframe
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        results = []
        n_samples = len(df)
        
        if self.verbose:
            logger.info(f"Processing {n_samples} samples in batches of {batch_size}...")
        
        for i in range(0, n_samples, batch_size):
            batch = df.iloc[i:i+batch_size]
            batch_results = self.predict(batch, source=source, return_input=False)
            results.append(batch_results)
            
            if self.verbose and i % (batch_size * 10) == 0:
                logger.info(f"Processed {min(i+batch_size, n_samples)}/{n_samples} samples")
        
        return pd.concat(results, ignore_index=True)


def run_prediction(
    config: Config,
    input_data: Union[pd.DataFrame, dict, str],
    mode: Literal["dev", "production"] = "dev",
    model_path: Optional[str] = None
) -> Union[pd.DataFrame, Dict[str, Any]]:
    """
    Convenience function to run predictions with a single call.
    
    Args:
        config: Configuration object
        input_data: Input data for prediction
        mode: Execution mode
        model_path: Optional path to model (overrides config)
        
    Returns:
        Prediction results
    """
    predictor = ModelPredictor(config, mode=mode)
    predictor.load_model(model_path)

    # Resolve data source from top-level config.data_source (fallback to 'local')
    source = getattr(config, 'data_source', 'local')
    return predictor.predict(input_data, source=source)


