import numpy as np
import pandas as pd

def detect_change_points_cusum(series: pd.Series) -> list:
    """
    Implements a Cumulative Sum (CUSUM) test to detect structural breaks.
    Returns the indices where a structural break is detected.
    (Simplified heuristic for Bayesian Change Point Detection).
    """
    series = series.dropna()
    mean = series.mean()
    cusum = (series - mean).cumsum()
    
    # Simple thresholding for demonstration.
    # A true Bayesian implementation requires MCMC or dynamic programming (e.g., ruptures).
    threshold = 2.0 * series.std() * np.sqrt(len(series))
    
    breaks = []
    # Find points where CUSUM exceeds the threshold
    if cusum.abs().max() > threshold:
        # The break point is typically where the CUSUM process hits its maximum absolute value
        break_idx = cusum.abs().idxmax()
        breaks.append(break_idx)
        
    return breaks
