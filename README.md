# Fertilizer Prediction – Playground Series S5E6

## Project Overview

This project predicts up to three optimal fertilizers for crops based on soil and environmental features. It uses the Kaggle Playground Series – Season 5, Episode 6: Predicting Optimal Fertilizers dataset. The end goal is to build a pipeline and API that can generate fertilizer recommendations, supporting precision agriculture and crop yield optimization.

## Dataset
- Kaggle competition: Playground Series S5E6
- The dataset is not included in this repo due to size and licensing.
- Place downloaded CSV files in the data/ folder before running scripts or notebooks.

## Usage
- EDA: Run notebooks in notebooks/eda.ipynb for initial data exploration.
- Modeling: Use scripts in src/ for preprocessing, feature engineering, and model training.
- API: TBD (future deployment pipeline).

## VENV
We use a virtual environment (`venv`) to isolate project dependencies.

| OS / Terminal | Activation Command |
| :--- | :--- |
| **macOS / Linux** | `source .venv/bin/activate` |
| **Windows (PowerShell)** | `.\.venv\Scripts\Activate.ps1` |
| **Windows (CMD)** | `.\.venv\Scripts\activate.bat` |

```bash
# Create the environment named .venv
python3 -m venv .venv

# Activate the environment (Example for Linux/macOS)
source .venv/bin/activate

# Install all listed packages and their exact versions
pip install -r requirements.txt

# Start local web server to view local experiments in MLFlow UI
mlflow ui

## Team
Arkojit Ghosh 
Yu-Hua Chen
David Goh 
Aliza Tarakanov 
