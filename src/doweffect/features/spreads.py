import pandas as pd
import numpy as np

def compute_intermarket_spreads(df: pd.DataFrame, energy_df: pd.DataFrame, metal_df: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """
    Computes strict t-1 safe Intermarket Spread features (Energy vs Metal).
    """
    df = df.copy()
    
    # Align dates
    spread_df = pd.DataFrame(index=energy_df.index)
    spread_df['Energy'] = energy_df['Close']
    spread_df['Metal'] = metal_df['Close']
    
    # Ratio
    spread_df['Ratio'] = spread_df['Energy'] / spread_df['Metal']
    
    # Log Spread Return
    spread_df['SpreadReturn'] = np.log(spread_df['Ratio']) - np.log(spread_df['Ratio'].shift(1))
    
    # Rolling Z-Score
    rolling_mean = spread_df['SpreadReturn'].rolling(window=window).mean()
    rolling_std = spread_df['SpreadReturn'].rolling(window=window).std()
    spread_df['SpreadZ'] = (spread_df['SpreadReturn'] - rolling_mean) / rolling_std
    
    # Categorization at t
    is_expansion = (spread_df['SpreadZ'] > 1.0).astype(int)
    is_contraction = (spread_df['SpreadZ'] < -1.0).astype(int)
    
    # LEAKAGE CONTROL: We strictly shift by 1.
    # The spread state for predicting day t must be the state computed at the end of t-1.
    df['is_spread_expansion'] = is_expansion.shift(1).reindex(df.index).fillna(0)
    df['is_spread_contraction'] = is_contraction.shift(1).reindex(df.index).fillna(0)
    
    return df
