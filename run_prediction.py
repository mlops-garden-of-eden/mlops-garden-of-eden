"""
Entry point script for running model predictions.
Supports both interactive and batch prediction modes.
"""
import argparse
import sys
import json
import pandas as pd
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_manager import get_config
from src.predictor import ModelPredictor, run_prediction
from src.utils import logger
from src.data_loader import load_prediction_data


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run predictions using trained ML models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single sample from JSON
  python scripts/run_prediction.py --input '{"feature1": 10, "feature2": "A"}' --mode dev
  
  # Batch prediction from CSV
  python scripts/run_prediction.py --input data/test.csv --mode production --output results.csv
  
  # Using specific model
  python scripts/run_prediction.py --input data/test.csv --model models/xgboost_model.pkl
  
  # Interactive mode
  python scripts/run_prediction.py --interactive
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config_base.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        help='Input data: JSON string, file path, or omit to use config path'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=False,
        help='Output file path for saving predictions (CSV or JSON). REQUIRED.'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        help='Path to model file (overrides config)'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['dev', 'production'],
        default='dev',
        help='Execution mode: dev (verbose, probabilities) or production (optimized)'
    )
    
    parser.add_argument(
        '--source',
        type=str,
        choices=['local', 'databricks'],
        default='local',
        help='Data source type'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1000,
        help='Batch size for large datasets (production mode)'
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode for testing'
    )
    
    parser.add_argument(
        '--return-input',
        action='store_true',
        help='Include input features in output'
    )
    
    return parser.parse_args()


def parse_input(input_str: str):
    """
    Parse input string as JSON or file path.
    
    Args:
        input_str: Input string
        
    Returns:
        Parsed input data
    """
    # Try as file path
    if Path(input_str).exists():
        return input_str
    
    # Try as JSON
    try:
        return json.loads(input_str)
    except json.JSONDecodeError:
        raise ValueError(
            f"Input must be a valid file path or JSON string. Got: {input_str}"
        )


def interactive_mode(predictor: ModelPredictor):
    """
    Run predictions in interactive mode.
    
    Args:
        predictor: Initialized ModelPredictor instance
    """
    print("\n" + "="*60)
    print("Interactive Prediction Mode")
    print("="*60)
    print("\nEnter feature values as JSON or 'quit' to exit.")
    print("Example: {\"feature1\": 10, \"feature2\": \"A\"}\n")
    
    if predictor.metadata.get('feature_names'):
        print(f"Expected features: {predictor.metadata['feature_names']}\n")
    
    while True:
        try:
            user_input = input(">>> ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Exiting interactive mode.")
                break
            
            if not user_input:
                continue
            
            # Parse input
            sample = json.loads(user_input)
            
            # Run prediction
            result = predictor.predict(sample, source="local", return_input=True)
            
            # Display result
            print("\nPrediction Result:")
            print("-" * 40)
            print(json.dumps(result, indent=2, default=str))
            print("-" * 40 + "\n")
            
        except json.JSONDecodeError:
            print("Error: Invalid JSON format. Please try again.\n")
        except Exception as e:
            print(f"Error: {e}\n")


def main():
    """Main execution function."""
    args = parse_args()
    
    try:
        # Load configuration
        config = get_config(args.config)

        # Setup logging (console + file) using config
        log_settings = getattr(getattr(config, 'tracking', None), 'log_settings', None)
        if log_settings is not None:
            from src.utils import setup_logging
            setup_logging(log_settings.level, log_settings.file_path)

        # Initialize predictor
        predictor = ModelPredictor(config, mode=args.mode)

        # Load model: prefer CLI override, else look in config.prediction.model_path if present
        pred_cfg = getattr(config, 'prediction', None)
        model_path = args.model if args.model else (getattr(pred_cfg, 'model_path', None) if pred_cfg is not None else None)
        if model_path is None:
            raise ValueError("Model path must be provided either via --model or in the config under prediction.model_path")
        predictor.load_model(model_path)

        # Interactive mode
        if args.interactive:
            interactive_mode(predictor)
            return

        # Prepare input data
        if args.input:
            input_data = parse_input(args.input)
        else:
            # Use configured data path
            input_data = None
            logger.info("No input specified. Using configured prediction data path.")


        # Determine output file path: CLI arg overrides config, error if neither
        pred_cfg = getattr(config, 'prediction', None)
        output_path = args.output
        if not output_path and pred_cfg is not None:
            output_path = getattr(pred_cfg, 'output_path', None)
        if not output_path:
            logger.error("You must specify an --output file path or set prediction.output_path in config. Console output is not supported.")
            print("Error: You must specify an --output file path or set prediction.output_path in config. Console output is not supported.")
            sys.exit(2)

        logger.info(f"Running predictions in {args.mode} mode...")

        # Log which prediction path is taken
        if isinstance(input_data, (str, type(None))) and args.mode == "production":
            logger.info("Prediction path: batch mode (production)")
            if input_data is None:
                logger.info("No input_data provided, loading prediction data from config.data")
                df = load_prediction_data(config.data, source=args.source)
            else:
                logger.info(f"Loading input data from file: {input_data}")
                df = pd.read_csv(input_data)

            logger.info(f"Loaded DataFrame for batch prediction with shape {df.shape}")
            results = predictor.predict_batch(
                df,
                batch_size=args.batch_size,
                source=args.source
            )
        else:
            logger.info("Prediction path: standard (single sample or dev mode)")
            results = predictor.predict(
                input_data,
                source=args.source,
                return_input=args.return_input
            )

        # Save results to output file (CSV for DataFrame, JSON for dict)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(results, dict):
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Results saved to: {output_path}")
        else:
            results.to_csv(output_path, index=False)
            logger.info(f"Results saved to: {output_path}")

        logger.info("Prediction pipeline completed successfully.")

    except Exception as e:
        logger.error(f"Prediction pipeline failed: {e}")
        if args.mode == "dev":
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()