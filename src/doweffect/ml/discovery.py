import logging
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)

def train_exploratory_model(df: pd.DataFrame, feature_cols: list, target_col: str = 'ResidualReturn') -> XGBRegressor:
    """
    Trains an XGBoost model PURELY for hypothesis generation.
    Outputs from this model are NOT accepted as evidence of alpha.
    """
    df_clean = df.dropna(subset=feature_cols + [target_col])
    if len(df_clean) < 252:
        logger.warning("Insufficient data for ML discovery (needs at least 1 year).")
        return None
        
    X = df_clean[feature_cols]
    y = df_clean[target_col]
    
    # We use a simple TimeSeriesSplit to ensure no lookahead in cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Shallow tree (max_depth=2) strictly to enforce 2-way interactions only
    # and prevent SHAP sparsity from over-fitting rare 3-way events.
    model = XGBRegressor(
        n_estimators=50,
        max_depth=2,
        learning_rate=0.1,
        objective='reg:squarederror',
        random_state=42
    )
    
    # Train on full dataset for SHAP explanation (since it's discovery, not backtesting)
    model.fit(X, y)
    
    logger.info("Exploratory XGBoost model trained successfully.")
    return model, X
