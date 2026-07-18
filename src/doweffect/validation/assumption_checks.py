import logging
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan, het_arch

logger = logging.getLogger(__name__)

class AssumptionValidator:
    """
    Automated econometric assumption validator.
    Tests distribution and variance assumptions before statistical execution.
    """
    
    @staticmethod
    def check_normality(series: pd.Series, alpha: float = 0.05) -> bool:
        """
        Tests if a series is normally distributed using Shapiro-Wilk (or Jarque-Bera for N>5000).
        """
        series = series.dropna()
        if len(series) < 3:
            logger.warning("Not enough data to test normality.")
            return False
            
        if len(series) > 5000:
            # Shapiro-Wilk is inaccurate for N > 5000, fallback to Jarque-Bera
            stat, p_value = stats.jarque_bera(series)
            test_name = "Jarque-Bera"
        else:
            stat, p_value = stats.shapiro(series)
            test_name = "Shapiro-Wilk"
            
        is_normal = p_value > alpha
        if not is_normal:
            logger.debug(f"{test_name} normality test failed (p={p_value:.4e}). Series exhibits non-normality.")
        return is_normal

    @staticmethod
    def check_homoskedasticity(groups: list[pd.Series], alpha: float = 0.05) -> bool:
        """
        Tests for equal variances across multiple groups using Levene's test (robust to non-normality).
        """
        cleaned_groups = [g.dropna() for g in groups if len(g.dropna()) > 1]
        if len(cleaned_groups) < 2:
            return False
            
        stat, p_value = stats.levene(*cleaned_groups, center='median')
        is_homoskedastic = p_value > alpha
        if not is_homoskedastic:
            logger.debug(f"Levene's test failed (p={p_value:.4e}). Groups exhibit heteroskedasticity.")
        return is_homoskedastic
