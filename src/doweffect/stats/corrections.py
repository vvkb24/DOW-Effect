import numpy as np
import pandas as pd

def westfall_young_permutation(group1_values: np.ndarray, group2_values: np.ndarray, n_permutations: int = 1000) -> float:
    """
    Implements a simplified Westfall-Young step-down permutation test for a single pairwise comparison.
    (Note: True Westfall-Young for Family-Wise Error Rate controls across ALL tests simultaneously.
    This function computes the exact permutation p-value for two groups which preserves the exact 
    data distribution, acting as the foundation for the FWER step-down procedure.)
    
    Args:
        group1_values: Values of group 1.
        group2_values: Values of group 2.
        n_permutations: Number of permutations.
        
    Returns:
        The permutation-based p-value.
    """
    # Observed absolute difference in means
    obs_diff = np.abs(np.mean(group1_values) - np.mean(group2_values))
    
    combined = np.concatenate([group1_values, group2_values])
    n1 = len(group1_values)
    
    count_exceed = 0
    for _ in range(n_permutations):
        # Permute the combined array
        np.random.shuffle(combined)
        
        # Compute mean difference on permuted arrays
        perm_diff = np.abs(np.mean(combined[:n1]) - np.mean(combined[n1:]))
        
        if perm_diff >= obs_diff:
            count_exceed += 1
            
    # p-value is the fraction of permutations that yielded a test statistic at least as extreme
    return count_exceed / n_permutations

def apply_fdr_correction(p_values: pd.Series) -> pd.Series:
    """
    Applies Benjamini-Hochberg False Discovery Rate (FDR) correction.
    While WY is primary, FDR is computed for comparison.
    """
    from statsmodels.stats.multitest import multipletests
    _, p_adj, _, _ = multipletests(p_values, method='fdr_bh')
    return pd.Series(p_adj, index=p_values.index)
