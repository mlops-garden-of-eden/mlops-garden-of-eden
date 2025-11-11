"""
Reusable feature engineering logic for both training and inference.
"""
from typing import Optional, List
import pandas as pd
from types import SimpleNamespace

def apply_feature_engineering(df: pd.DataFrame, config) -> pd.DataFrame:
    """
    Applies feature engineering operations as defined in config.feature_engineering.operations.
    Args:
        df: Input DataFrame (will not be mutated)
        config: Config object with feature_engineering.operations
    Returns:
        DataFrame with engineered features added (copy)
    """
    fe_cfg = getattr(config, "feature_engineering", None)
    if not fe_cfg or not getattr(fe_cfg, "enable", False):
        return df.copy()

    operations = getattr(fe_cfg, "operations", []) or []
    if not operations:
        return df.copy()

    df_fe = df.copy()
    for op in operations:
        output_col = getattr(op, 'output', None) or op.get('output')
        formula = getattr(op, 'formula', None) or op.get('formula')
        if not output_col or not formula:
            continue
        try:
            df_fe[output_col] = df_fe.eval(formula)
        except Exception:
            # Optionally log or warn here
            pass
    return df_fe
