# Fertilizer Prediction Project

## Project Overview

This project predicts up to three optimal fertilizers for crops based on soil and environmental features. It uses the Kaggle Playground Series – Season 5, Episode 6: Predicting Optimal Fertilizers dataset. The goal is to build a machine learning pipeline and interactive tools that generate fertilizer recommendations, supporting precision agriculture and crop yield optimization.

## Dataset

- **Source**: Kaggle competition - Playground Series S5E6
- **Note**: The dataset is not included in this repository due to size and licensing restrictions
- **Setup**: Download the CSV files from Kaggle and place them in the `data/` folder before running scripts or notebooks

## Setup

### Virtual Environment

We use a virtual environment (`venv`) to isolate project dependencies.

```bash
# Create the environment named .venv
python3 -m venv .venv
```

**Activate the environment:**

| OS / Terminal | Activation Command |
| :--- | :--- |
| **macOS / Linux** | `source .venv/bin/activate` |
| **Windows (PowerShell)** | `.\.venv\Scripts\Activate.ps1` |
| **Windows (CMD)** | `.\.venv\Scripts\activate.bat` |

### Install Dependencies

Install all required packages (including ML libraries, YAML support, and web app dependencies):

```bash
pip install -r requirements.txt
```

### MLflow UI (Optional)

To view local experiments in the MLflow UI:

```bash
mlflow ui
```

## Configuration

All configuration is managed through YAML files in the `config/` directory:

- **`config_base.yaml`**: Base/default settings
- **`config_dev.yaml`**: Development settings (shows probabilities, verbose logging)
- **`config_prod.yaml`**: Production settings (no probabilities, minimal output)

### Key Configuration Sections

- **`data`**: Data paths, features, and metadata columns
- **`tuning`**: Model selection and hyperparameters
- **`artifacts`**: Model artifact saving options
- **`prediction`**: Prediction options (model path, output path, return_probabilities)

## Usage

### Exploratory Data Analysis (EDA)

Run the Jupyter notebooks for initial data exploration:

```bash
jupyter notebook notebooks/eda.ipynb
```

### Training (Experiment Pipeline)

Train models and save artifacts using a configuration file:

```bash
python run_experiments.py --config CONFIG_PATH [--data-source {local|databricks}] [--env ENV_LABEL]
```

**Arguments:**
- `--config`: Path to the YAML config file (default: `config/config_base.yaml`)
- `--data-source`: Optional override for data source (`local` or `databricks`). Overrides config value.
- `--env`: Environment label passed to the runner (default: `dev`)

**Output**: Artifacts (pipeline, label encoder) are saved as `.pkl` files in the configured directory. MLflow is used for experiment tracking.

### Prediction (Batch & Interactive)

Run predictions on a dataset or single sample using a trained model:

```bash
python run_prediction.py [--config CONFIG_PATH] [--input INPUT] [--output OUTPUT] [--model MODEL_PATH] [--mode {dev|production}] [--source {local|databricks}] [--batch-size N] [--interactive] [--return-input]
```

**Arguments:**
- `--config`: Path to configuration file (default: `config/config_base.yaml`)
- `--input`: Input data (JSON string, file path, or omit to use config path)
- `--output`: Output file path for saving predictions (CSV or JSON)
- `--model`: Path to model file (overrides config)
- `--mode`: Execution mode: `dev` (verbose, probabilities) or `production` (optimized)
- `--source`: Data source type (`local` or `databricks`)
- `--batch-size`: Batch size for large datasets (default: 1000)
- `--interactive`: Run in interactive mode for testing
- `--return-input`: Include input features in output

**Examples:**

Single sample prediction from JSON:
```bash
python run_prediction.py --input '{"feature1": 10, "feature2": "A"}' --mode dev
```

Batch prediction from CSV:
```bash
python run_prediction.py --input data/test.csv --mode production --output results.csv
```

Using a specific model:
```bash
python run_prediction.py --input data/test.csv --model models/xgboost_model.pkl
```

Interactive mode:
```bash
python run_prediction.py --interactive
```

### Web Application

Launch the Streamlit web app for interactive predictions:

```bash
streamlit run app.py
```

**Features:**
- By default, uses `config/config_prod.yaml` for production-style predictions
- Enter feature values in the form to get predicted fertilizer names
- To use a different config, edit `DEFAULT_CONFIG_PATH` in `app.py`

## Testing

### Local Testing

Run local tests:

```bash
python run_test.py
```

### Remote Testing (Databricks)

First, install and configure the Databricks CLI:

```bash
# Install Databricks CLI
curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh

# Configure credentials
export DATABRICKS_HOST=https://dbc-6713e564-9737.cloud.databricks.com
export DATABRICKS_TOKEN=<your_token_here>

# Validate and deploy
databricks bundle validate
databricks bundle deploy
```

Then run the test job on Databricks:

```bash
databricks bundle run Test
```

**Note**: Remote tests can also be triggered automatically during push (if configured in CI/CD).

## Project Structure

```
├── config/              # Configuration files (YAML)
├── data/                # Dataset files (not included in repo)
├── notebooks/           # Jupyter notebooks for EDA
├── src/                 # Source code for preprocessing, feature engineering, and modeling
├── models/              # Saved model artifacts (.pkl files)
├── app.py               # Streamlit web application
├── run_experiments.py   # Training pipeline script
├── run_prediction.py    # Prediction script
├── run_test.py          # Local testing script
└── requirements.txt     # Python dependencies
```

## Future Work

- API deployment pipeline (TBD)
- Additional model architectures
- Enhanced feature engineering
- Production deployment infrastructure