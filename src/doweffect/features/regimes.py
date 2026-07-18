import pandas as pd
import numpy as np

def compute_regimes(df: pd.DataFrame, vol_window: int = 21, trend_window: int = 50) -> pd.DataFrame:
    """
    Computes strict t-1 safe regime flags.
    No forward-looking data is used.
    """
    df = df.copy()
    
    # Ensure High/Low are available for True Range. If not, use proxy.
    if 'High' in df.columns and 'Low' in df.columns:
        # True Range: max(H-L, |H-Cp|, |L-Cp|)
        prev_close = df['Close'].shift(1)
        tr1 = df['High'] - df['Low']
        tr2 = (df['High'] - prev_close).abs()
        tr3 = (df['Low'] - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=vol_window).mean()
        # Normalized ATR
        natr = atr / df['Close']
    else:
        # Fallback to realized vol
        natr = df['Return'].rolling(window=vol_window).std()
        
    # Strictly shift by 1 to ensure t-1 safety
    t1_natr = natr.shift(1)
    
    # Define High Volatility Regime: Top 25% of rolling 252-day history
    rolling_80th_natr = t1_natr.rolling(window=252).quantile(0.80)
    df['regime_high_vol'] = (t1_natr > rolling_80th_natr).astype(int)
    
    # Trend Regime: Close vs MA
    ma_trend = df['Close'].rolling(window=trend_window).mean()
    t1_ma_trend = ma_trend.shift(1)
    df['regime_bull_trend'] = (df['Close'].shift(1) > t1_ma_trend).astype(int)
    
    # Liquidity Regime (Amihud)
    # Using existing Amihud but shifting to ensure t-1 safety
    if 'Volume' in df.columns:
        rupee_volume = df['Volume'] * df['Close']
        illiquidity = df['Return'].abs() / rupee_volume.replace(0, np.nan) * 1e6
        avg_illiquidity = illiquidity.rolling(window=21).mean()
        t1_illiquidity = avg_illiquidity.shift(1)
        
        rolling_80th_illiq = t1_illiquidity.rolling(window=252).quantile(0.80)
        df['regime_low_liquidity'] = (t1_illiquidity > rolling_80th_illiq).astype(int)
    else:
        df['regime_low_liquidity'] = 0
        
    return df
