import pandas as pd

def compute_persistence_score(rolling_p_values: pd.Series, alpha: float = 0.05) -> float:
    """
    Computes the Effect Persistence Score.
    Definition: Fraction of rolling windows in which the anomaly 
    survives multiple-testing correction (p < alpha).
    
    Args:
        rolling_p_values: Series of corrected p-values from rolling windows.
        alpha: Significance threshold.
        
    Returns:
        float: Persistence score between 0.0 and 1.0.
    """
    if rolling_p_values.empty:
        return 0.0
        
    significant_windows = (rolling_p_values < alpha).sum()
    total_windows = len(rolling_p_values)
    
    return significant_windows / total_windows
