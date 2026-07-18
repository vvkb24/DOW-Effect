import pandas as pd
import numpy as np

def compute_raw_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes raw unadjusted log returns.
    Assumes df contains 'Close' and 'Dividends' columns from yfinance.
    We add discrete dividends back on ex-dates to prevent backward smearing
    from adjusted prices.
    """
    if 'Close' not in df.columns:
        raise ValueError("DataFrame must contain a 'Close' column.")
        
    df = df.copy()
    
    # Calculate simple close-to-close return
    # If there's a dividend today, the total return includes the dividend amount 
    # relative to yesterday's close.
    prev_close = df['Close'].shift(1)
    
    dividend = df['Dividends'] if 'Dividends' in df.columns else 0.0
    
    # Total return = (Close + Dividend - Prev_Close) / Prev_Close
    df['Return'] = (df['Close'] + dividend - prev_close) / prev_close
    
    # Log return = log(1 + Return)
    df['LogReturn'] = np.log1p(df['Return'])
    
    # Overnight Gap: (Open - Prev_Close) / Prev_Close
    if 'Open' in df.columns:
        df['OvernightReturn'] = (df['Open'] - prev_close) / prev_close
        df['IntradayReturn'] = (df['Close'] - df['Open']) / df['Open']
        
    # Standardize day of week: 0=Monday, 4=Friday
    df['DayOfWeek'] = df.index.dayofweek
    df['Week'] = df.index.isocalendar().week
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    
    return df
