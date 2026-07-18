import pandas as pd
import numpy as np
from scipy import stats
import itertools
from typing import Dict, List

def exhaustive_pairwise_tests(df: pd.DataFrame, value_col: str = 'Return') -> pd.DataFrame:
    """
    Performs exhaustive pairwise hypothesis tests (Welch t-test and Mann-Whitney U) 
    across all combinations of weekdays.
    
    Args:
        df: DataFrame containing the value column and a 'DayOfWeek' column (0=Mon, 4=Fri).
        value_col: The column to test (e.g. 'Return', 'ResidualReturn').
        
    Returns:
        DataFrame of test statistics and raw p-values.
    """
    days = range(5)
    results = []
    
    for day1, day2 in itertools.combinations(days, 2):
        group1 = df[df['DayOfWeek'] == day1][value_col].dropna()
        group2 = df[df['DayOfWeek'] == day2][value_col].dropna()
        
        if len(group1) < 10 or len(group2) < 10:
            continue
            
        # Welch t-test (unequal variance assumed)
        t_stat, t_pval = stats.ttest_ind(group1, group2, equal_var=False)
        
        # Mann-Whitney U test (non-parametric)
        u_stat, u_pval = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        
        # Difference in means
        mean_diff = group1.mean() - group2.mean()
        
        results.append({
            'Day1': day1,
            'Day2': day2,
            'MeanDiff': mean_diff,
            'Welch_t': t_stat,
            'Welch_p': t_pval,
            'MWU_stat': u_stat,
            'MWU_p': u_pval
        })
        
    return pd.DataFrame(results)
