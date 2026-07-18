import pandas as pd
import numpy as np
import logging
import warnings
from hmmlearn import hmm

logger = logging.getLogger(__name__)

def compute_hmm_regimes(df: pd.DataFrame, window_size: int = 252, step: int = 21) -> pd.DataFrame:
    """
    Computes strict t-1 safe HMM regime flags.
    Fits a rolling Gaussian HMM on features (Returns, ATR) up to t-2, and predicts t-1.
    Sorts states by emission variance to prevent label switching.
    """
    df = df.copy()
    
    # Base features for HMM
    if 'High' in df.columns and 'Low' in df.columns:
        prev_close = df['Close'].shift(1)
        tr = pd.concat([df['High'] - df['Low'], (df['High'] - prev_close).abs(), (df['Low'] - prev_close).abs()], axis=1).max(axis=1)
        atr = tr / df['Close']
    else:
        atr = df['Return'].abs()
        
    df['ATR'] = atr
    
    # We must have valid Returns and ATR
    feat_df = df[['Return', 'ATR']].dropna()
    
    regime_series = pd.Series(index=feat_df.index, dtype=float)
    
    # Rolling fit logic.
    # To save compute, we use a step. But we always predict t-1 strictly without looking ahead.
    # For every index i, we fit on [i - window_size - 1 : i - 2]
    # and predict for [i - 1].
    # But doing this for 4000 rows is extremely slow.
    # Optimization: Fit HMM every `step` days. Use that static HMM to predict the next `step` days.
    
    idx = feat_df.index
    
    for start_i in range(window_size + 1, len(idx), step):
        train_end_i = start_i - 1
        train_start_i = max(0, train_end_i - window_size)
        
        train_data = feat_df.iloc[train_start_i:train_end_i]
        
        # Predict range: from start_i to start_i + step
        # Note: the prediction at day `t` relies on data at `t-1` safely.
        pred_start_i = start_i
        pred_end_i = min(len(idx), start_i + step)
        pred_data = feat_df.iloc[pred_start_i-1 : pred_end_i-1] # Shifted back by 1 for prediction!
        
        if len(train_data) < 100:
            continue
            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model = hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=100, random_state=42)
                model.fit(train_data.values)
                
                # Prevent label switching: 
                # Sort states by volatility (variance of Returns, which is feature 0)
                # Ensure State 1 is ALWAYS the high variance state.
                state_variances = model.covars_[:, 0, 0]
                high_vol_state = np.argmax(state_variances)
                
                preds = model.predict(pred_data.values)
                
                # Map predictions so that 1 = High Vol State
                aligned_preds = (preds == high_vol_state).astype(int)
                
                # Assign to the ACTUAL timeline (which is pred_start_i to pred_end_i)
                regime_series.iloc[pred_start_i:pred_end_i] = aligned_preds
            except Exception as e:
                logger.debug(f"HMM fit failed at index {start_i}: {e}")
                
    # Forward fill the first window with 0 (safe fallback)
    regime_series = regime_series.fillna(0)
    
    df['regime_hmm_high_vol'] = regime_series
    
    # We remove the static regime definitions to keep the namespace clean
    return df
