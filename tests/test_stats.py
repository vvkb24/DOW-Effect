import numpy as np
import pandas as pd
import pytest
from doweffect.stats.time_series import TimeSeriesValidator
from doweffect.stats.bootstrap import compute_optimal_block_length, moving_block_bootstrap
from doweffect.stats.corrections import westfall_young_permutation

def test_ljung_box():
    # Generate random normal (white noise) - should not have autocorrelation
    np.random.seed(42)
    white_noise = pd.Series(np.random.normal(0, 1, 100))
    res = TimeSeriesValidator.test_autocorrelation(white_noise, lags=5)
    assert not res['has_autocorr'], "White noise should not fail Ljung-Box"
    
    # Generate AR(1) process
    ar1 = [0.0]
    for _ in range(99):
        ar1.append(0.8 * ar1[-1] + np.random.normal(0, 1))
    res_ar = TimeSeriesValidator.test_autocorrelation(pd.Series(ar1), lags=5)
    assert res_ar['has_autocorr'], "AR(1) process should fail Ljung-Box"

def test_mbb_block_length():
    np.random.seed(42)
    white_noise = pd.Series(np.random.normal(0, 1, 1000))
    b_opt = compute_optimal_block_length(white_noise)
    assert b_opt == 1, "White noise should have block length 1"
    
    # AR(1) with high positive correlation
    ar1 = [0.0]
    for _ in range(999):
        ar1.append(0.8 * ar1[-1] + np.random.normal(0, 1))
    b_opt_ar = compute_optimal_block_length(pd.Series(ar1))
    assert b_opt_ar > 1, "Highly autocorrelated series should have block length > 1"

def test_moving_block_bootstrap():
    np.random.seed(42)
    series = pd.Series(np.arange(10))
    bootstrapped = moving_block_bootstrap(series, n_bootstraps=5, block_length=3)
    assert bootstrapped.shape == (5, 10)
    # Check that it's drawing from the original values
    assert np.all(np.isin(bootstrapped, series.values))

def test_westfall_young():
    np.random.seed(42)
    # Two identical distributions
    g1 = np.random.normal(0, 1, 50)
    g2 = np.random.normal(0, 1, 50)
    pval = westfall_young_permutation(g1, g2, n_permutations=100)
    assert pval > 0.05, "Identical distributions should not be significant"
    
    # Two distinctly different distributions
    g3 = np.random.normal(5, 1, 50)
    pval2 = westfall_young_permutation(g1, g3, n_permutations=100)
    assert pval2 < 0.05, "Different distributions should be significant"
