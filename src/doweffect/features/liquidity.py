import pandas as pd
import numpy as np

def compute_amihud_illiquidity(df: pd.DataFrame, window: int = 21) -> pd.Series:
    """
    Computes the Amihud (2002) Illiquidity Ratio.
    Defined as the rolling average of |Return| / (Volume * Price).
    Note: Volume in Indian markets is often large, resulting in very small ratios.
    We multiply by 10^6 for scale.
    """
    if 'Volume' not in df.columns or 'Close' not in df.columns or 'Return' not in df.columns:
        raise ValueError("Missing columns for Amihud calculation.")
        
    rupee_volume = df['Volume'] * df['Close']
    
    # Avoid division by zero
    rupee_volume = rupee_volume.replace(0, np.nan)
    
    daily_illiquidity = df['Return'].abs() / rupee_volume * 1e6
    return daily_illiquidity.rolling(window=window).mean()

def compute_corwin_schultz_spread(df: pd.DataFrame, window: int = 21) -> pd.Series:
    """
    Computes Corwin-Schultz (2012) bid-ask spread proxy from High/Low prices.
    Simplified version.
    """
    if 'High' not in df.columns or 'Low' not in df.columns:
        raise ValueError("Missing High/Low columns for CS spread.")
        
    hl_ratio = np.log(df['High'] / df['Low']) ** 2
    gamma = hl_ratio.rolling(window=2).sum()
    beta = np.log(df['High'].rolling(window=2).max() / df['Low'].rolling(window=2).min()) ** 2
    
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / (3 - 2 * np.sqrt(2)) - np.sqrt(gamma / (3 - 2 * np.sqrt(2)))
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    
    # Floor at zero
    spread = spread.clip(lower=0)
    return spread.rolling(window=window).mean()
