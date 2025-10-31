import argparse
import logging
import os
from src.utils import setup_logging, logger
from src.config_manager import get_config
from src.experiments_runner import run_experiment_stage


def main():
    parser = argparse.ArgumentParser(description="Run MLOps experiments")
    parser.add_argument(
        "--config",
        default="config/config_base.yaml",
        help="Path to the base YAML config file (relative to repo root or absolute)"
    )
    parser.add_argument(
        "--data-source",
        default=None,
        help="Optional override for data_source (local|databricks). If provided, overrides config value."
    )
    parser.add_argument(
        "--env",
        default="dev",
        help="Environment label passed to the runner (default: dev)"
    )

    args = parser.parse_args()

    # Load Configuration
    config = get_config(base_path=args.config)

    # Allow override by environment variable as well
    env_override = os.environ.get("DATA_SOURCE")
    if args.data_source:
        config.data_source = args.data_source
    elif env_override:
        config.data_source = env_override

    # Initialize Logging
    setup_logging(
        level=getattr(logging, config.tracking.log_settings.level.upper()),
        log_file=config.tracking.log_settings.file_path
    )

    # Define Environment-Specific Parameters
    ENV = args.env

    # Start Pipeline Execution
    logger.info("--- MLOps Pipeline Start ---")

    try:
        final_run_id = run_experiment_stage(config, ENV)
        logger.info(f"Pipeline finished successfully. Best run ID: {final_run_id}")
    except Exception as e:
        logger.critical(f"Pipeline terminated with a critical failure.", exc_info=True)


if __name__ == "__main__":
    main()
