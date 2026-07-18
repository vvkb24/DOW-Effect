import numpy as np
import pandas as pd

def compute_optimal_block_length(series: pd.Series) -> int:
    """
    Computes the optimal block length for Moving Block Bootstrap 
    using a simplified Politis & White (2004) heuristic based on 
    the first-order autocorrelation.
    """
    series = series.dropna()
    n = len(series)
    if n < 10:
        return 1
        
    # Autocorrelation at lag 1
    rho = series.autocorr(lag=1)
    
    # Politis-White heuristic approximation:
    # b_opt = (2 * rho / (1 - rho**2))**(2/3) * n**(1/3)
    # If rho is small or negative, block length defaults to small values
    if pd.isna(rho) or rho <= 0:
        return 1
        
    b_opt = ((2 * rho) / (1 - rho**2))**(2/3) * (n**(1/3))
    
    # Bound between 1 and sqrt(N)
    b_opt = max(1, min(int(np.ceil(b_opt)), int(np.sqrt(n))))
    return b_opt

def moving_block_bootstrap(series: pd.Series, n_bootstraps: int = 1000, block_length: int = None) -> np.ndarray:
    """
    Generates bootstrapped series using the Moving Block Bootstrap (MBB) method
    to preserve temporal dependence (autocorrelation).
    Returns a 2D array of shape (n_bootstraps, len(series)).
    """
    series_vals = series.dropna().values
    n = len(series_vals)
    
    if block_length is None:
        block_length = compute_optimal_block_length(pd.Series(series_vals))
        
    # Generate overlapping blocks
    blocks = [series_vals[i:i+block_length] for i in range(n - block_length + 1)]
    n_blocks_available = len(blocks)
    
    n_blocks_needed = int(np.ceil(n / block_length))
    
    bootstrapped_samples = np.empty((n_bootstraps, n))
    
    # Fix seed to globally enforced seed 42 (assumes np.random.seed(42) was set externally, 
    # but we can enforce it here via random state for pure functions if desired, 
    # though it's better to manage via global config. We'll use the default rng).
    for i in range(n_bootstraps):
        chosen_indices = np.random.randint(0, n_blocks_available, size=n_blocks_needed)
        # Concatenate blocks and truncate to exact length n
        sample = np.concatenate([blocks[idx] for idx in chosen_indices])[:n]
        bootstrapped_samples[i, :] = sample
        
    return bootstrapped_samples
