import pandas as pd
import numpy as np

def compute_annualized_alpha(residual_returns: pd.Series, trading_days_per_year: int = 252) -> float:
    """
    Computes annualized alpha from residual returns.
    Alpha is the mean of the residuals multiplied by trading days.
    """
    if residual_returns.empty:
        return 0.0
    daily_alpha = residual_returns.mean()
    annualized_alpha = daily_alpha * trading_days_per_year
    return float(annualized_alpha)

def compute_information_ratio(residual_returns: pd.Series, trading_days_per_year: int = 252) -> float:
    """
    Computes Information Ratio (IR).
    IR = Annualized Alpha / Annualized Tracking Error (residual volatility)
    """
    if residual_returns.empty:
        return 0.0
    
    daily_alpha = residual_returns.mean()
    daily_te = residual_returns.std()
    
    if daily_te == 0 or pd.isna(daily_te):
        return 0.0
        
    annualized_alpha = daily_alpha * trading_days_per_year
    annualized_te = daily_te * np.sqrt(trading_days_per_year)
    
    return float(annualized_alpha / annualized_te)
