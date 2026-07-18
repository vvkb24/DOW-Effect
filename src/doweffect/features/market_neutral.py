import pandas as pd
import statsmodels.api as sm

def compute_residual_returns(stock_returns: pd.Series, factor_df: pd.DataFrame) -> pd.Series:
    """
    Constructs a synthetic 3-factor model using Market, Size, and Sector proxies.
    Returns the residual returns (epsilon) from the OLS regression.
    
    Args:
        stock_returns: Series of daily stock log returns.
        factor_df: DataFrame containing factor returns (e.g., 'MKT', 'SIZE', 'SECTOR').
    """
    # Align data
    aligned = pd.concat([stock_returns.rename("stock"), factor_df], axis=1).dropna()
    if len(aligned) < 30:
        return pd.Series(index=stock_returns.index, dtype=float)
        
    y = aligned["stock"]
    X = sm.add_constant(aligned.drop(columns=["stock"]))
    
    model = sm.OLS(y, X).fit()
    residuals = model.resid
    
    # Reindex back to original shape, filling missing with NaN
    return residuals.reindex(stock_returns.index)
