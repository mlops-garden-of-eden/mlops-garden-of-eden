

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd

# --- DropColumnsTransformer ---
class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to drop specified columns from a DataFrame.
    """
    def __init__(self, columns=None):
        self.columns = columns if columns is not None else []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        cols_to_drop = [col for col in self.columns if col in X_.columns]
        return X_.drop(columns=cols_to_drop, errors='ignore')

# --- FeatureEngineeringTransformer ---
class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to apply feature engineering formulas from config.
    """
    def __init__(self, config=None):
        self.config = config

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        fe_cfg = getattr(self.config, "feature_engineering", None)
        if not fe_cfg or not getattr(fe_cfg, "enable", False):
            return df
        operations = getattr(fe_cfg, "operations", []) or []
        for op in operations:
            output_col = getattr(op, 'output', None) or op.get('output')
            formula = getattr(op, 'formula', None) or op.get('formula')
            if not output_col or not formula:
                continue
            try:
                df[output_col] = df.eval(formula)
            except Exception:
                pass
        return df

def create_preprocessor(numerical_features, categorical_features, meta_features=None, config=None):
    """
    Creates a scikit-learn Pipeline for preprocessing, including dropping meta features and feature engineering.
    """
    if meta_features is None:
        meta_features = []

    # Pipeline for numerical features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Pipeline for categorical features
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine transformers using ColumnTransformer
    col_transform = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    # Full pipeline: drop meta features, then feature engineering, then column transform
    pipeline = Pipeline([
        ('drop_meta', DropColumnsTransformer(columns=meta_features)),
        ('feature_engineering', FeatureEngineeringTransformer(config=config)),
        ('col_transform', col_transform)
    ])
    return pipeline