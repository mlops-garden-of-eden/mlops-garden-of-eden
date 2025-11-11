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
        Args:
            model_path: Path to saved model. If None, uses config.
        """
        if model_path is None:
            pred_cfg = getattr(self.config, 'prediction', None)
            model_path = getattr(pred_cfg, 'model_path', None) if pred_cfg is not None else None
            if model_path is None:
                logger.error("Model path not provided to load_model() and no 'prediction.model_path' found in config.")
                raise ValueError("Model path not provided to load_model() and no 'prediction.model_path' found in config.")

        model_path = Path(model_path)
        if not model_path.exists():
            logger.error(f"Model not found at: {model_path}")
            raise FileNotFoundError(f"Model not found at: {model_path}")

        logger.info(f"Loading model pipeline from: {model_path}")

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


