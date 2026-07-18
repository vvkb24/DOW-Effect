import numpy as np
from scipy.stats import norm

def compute_ex_ante_mde(n_samples: int, std_dev: float, alpha: float = 0.05, power: float = 0.80) -> float:
    """
    Computes the ex-ante Minimum Detectable Effect (MDE) for a two-sided test.
    This replaces tautological post-hoc power calculations.
    
    Args:
        n_samples: Number of observations.
        std_dev: Estimated standard deviation of the sample.
        alpha: Significance level (default 0.05).
        power: Desired statistical power (default 0.80).
        
    Returns:
        The absolute difference in means that can be detected with the given power.
    """
    if n_samples < 2 or std_dev <= 0:
        return np.nan
        
    z_alpha = norm.ppf(1 - alpha / 2)
    z_power = norm.ppf(power)
    
    mde = (z_alpha + z_power) * std_dev / np.sqrt(n_samples)
    return float(mde)
