import logging
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller, kpss

logger = logging.getLogger(__name__)

class TimeSeriesValidator:
    """
    Quantifies IID violations and structural dependencies in financial time series.
    """
    
    @staticmethod
    def test_autocorrelation(series: pd.Series, lags: int = 10, alpha: float = 0.05) -> dict:
        """
        Ljung-Box test for serial correlation.
        Returns True if the series has significant autocorrelation.
        """
        series = series.dropna()
        if len(series) < lags + 1:
            return {"has_autocorr": False, "p_value": 1.0}
            
        lb_results = acorr_ljungbox(series, lags=[lags], return_df=True)
        p_val = lb_results.iloc[0]['lb_pvalue']
        
        has_autocorr = bool(p_val < alpha)
        if has_autocorr:
            logger.debug(f"Ljung-Box test failed (p={p_val:.4e}). Series exhibits serial correlation.")
            
        return {"has_autocorr": has_autocorr, "p_value": float(p_val)}

    @staticmethod
    def test_volatility_clustering(series: pd.Series, lags: int = 10, alpha: float = 0.05) -> dict:
        """
        Engle's ARCH LM test for autoregressive conditional heteroskedasticity.
        """
        series = series.dropna()
        if len(series) < lags + 1:
            return {"has_arch": False, "p_value": 1.0}
            
        # The function het_arch returns (lm_stat, lm_pval, f_stat, f_pval)
        arch_test = het_arch(series, nlags=lags)
        p_val = arch_test[1]
        
        has_arch = bool(p_val < alpha)
        if has_arch:
            logger.debug(f"ARCH LM test failed (p={p_val:.4e}). Series exhibits volatility clustering.")
            
        return {"has_arch": has_arch, "p_value": float(p_val)}

    @staticmethod
    def test_stationarity(series: pd.Series, alpha: float = 0.05) -> dict:
        """
        Performs ADF and KPSS tests for stationarity.
        ADF H0: Non-stationary (unit root).
        KPSS H0: Stationary.
        """
        series = series.dropna()
        if len(series) < 20:
            return {"is_stationary": True, "adf_p": 0.0, "kpss_p": 1.0}
            
        adf_stat, adf_p, _, _, _, _ = adfuller(series, autolag='AIC')
        kpss_stat, kpss_p, _, _ = kpss(series, regression='c', nlags='auto')
        
        # Series is stationary if ADF rejects H0 AND KPSS fails to reject H0
        is_stationary = (adf_p < alpha) and (kpss_p >= alpha)
        
        if not is_stationary:
            logger.warning(f"Stationarity conflict. ADF p={adf_p:.4e}, KPSS p={kpss_p:.4e}.")
            
        return {
            "is_stationary": bool(is_stationary),
            "adf_p": float(adf_p),
            "kpss_p": float(kpss_p)
        }
