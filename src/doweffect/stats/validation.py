import logging
import pandas as pd
import numpy as np
import re
from doweffect.stats.panel import estimate_panel_effects

logger = logging.getLogger(__name__)

def walk_forward_validation(df: pd.DataFrame, formula: str, initial_train_years: int = 5, step_years: int = 1, seasonal_aware: bool = False) -> dict:
    """
    Performs out-of-sample walk-forward validation for a conditional alpha formula.
    
    Args:
        df: The dataset containing all features and targets.
        formula: The panel regression formula (e.g., 'ResidualReturn ~ day_0 * regime_high_vol')
        
    Returns:
        A dictionary with stability scores and OOS t-statistics for the interaction term.
    """
    if df.empty:
        return {"stable": False, "reason": "Empty DataFrame"}
        
    # Extract the pure interaction term name as patsy would report it
    # E.g., 'day_0 * regime_high_vol' -> 'day_0:regime_high_vol'
    try:
        interaction_term = formula.split('~')[1].strip().split('*')[0].strip() + ':' + formula.split('~')[1].strip().split('*')[1].strip()
    except IndexError:
        logger.warning(f"Formula '{formula}' does not contain a standard '*' interaction.")
        interaction_term = None
        
    if seasonal_aware:
        # Shift year boundary to July 1st.
        # Months 1-6 belong to (Year - 1). Months 7-12 belong to (Year).
        year_series = df.index.year - (df.index.month <= 6).astype(int)
    else:
        year_series = df.index.year

    years = sorted(year_series.unique())
    if len(years) < initial_train_years + step_years:
        logger.warning("Not enough years for walk-forward validation.")
        return {"stable": False, "reason": "Insufficient history"}
        
    oos_tstats = []
    oos_predictive_metrics = []
    
    for i in range(initial_train_years, len(years), step_years):
        train_years = years[:i]
        test_years = years[i:i+step_years]
        
        train_df = df[year_series.isin(train_years)]
        test_df = df[year_series.isin(test_years)]
        
        if train_df.empty or test_df.empty:
            continue
            
        try:
            # Estimate on training data
            train_res = estimate_panel_effects(train_df, formula)
            
            # Extract relevant columns and drop NAs for test_df
            cols = re.sub(r'[~+*:]', ' ', formula).split()
            cols = [c for c in cols if c in test_df.columns]
            valid_test_df = test_df.dropna(subset=cols)
            
            if valid_test_df.empty:
                continue

            # Predict on test data using train_res
            preds = train_res.predict(valid_test_df)
            
            # Align y_test to preds index (statsmodels drops NAs silently during predict)
            preds = preds.dropna()
            target_col = formula.split('~')[0].strip()
            y_test = valid_test_df.loc[preds.index, target_col]
            
            # Compute predictive OOS metrics
            errors = y_test - preds
            rmse = float(np.sqrt((errors ** 2).mean()))
            mae = float(errors.abs().mean())
            bias = float(errors.mean())
            sst = ((y_test - y_test.mean()) ** 2).sum()
            r2 = float(1 - ((errors ** 2).sum() / sst)) if sst != 0 else np.nan
            
            # We ONLY run the coefficient refit AFTER calculating the predictive metrics.
            # Institutional standard for structural inference is sub-period stability of the coefficient.
            test_res = estimate_panel_effects(valid_test_df, formula)
            print(f"DEBUG: interaction_term is '{interaction_term}'")
            print(f"DEBUG: test_res.tvalues.keys() are {list(test_res.tvalues.keys())}")
            
            if interaction_term and interaction_term in test_res.tvalues:
                t_stat = test_res.tvalues[interaction_term]
                oos_tstats.append(t_stat)
                
                oos_predictive_metrics.append({
                    "train_period": f"{min(train_years)}-{max(train_years)}",
                    "test_period": f"{min(test_years)}-{max(test_years)}",
                    "t_stat": t_stat,
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2,
                    "bias": bias
                })
                
                
        except Exception as e:
            logger.exception(f"Walk-forward failed for window {test_years}: {e}")
            
    if not oos_tstats:
        return {"stable": False, "reason": "Could not extract OOS t-stats"}
        
    # Consistency check: Does the t-stat maintain the same sign in > 70% of OOS windows?
    signs = [1 if t > 0 else -1 for t in oos_tstats]
    dominant_sign_ratio = max(signs.count(1), signs.count(-1)) / len(signs)
    
    is_stable = dominant_sign_ratio >= 0.70
    
    return {
        "stable": is_stable,
        "dominant_sign_ratio": dominant_sign_ratio,
        "oos_tstats": oos_tstats,
        "predictive_metrics": oos_predictive_metrics,
        "reason": "Stable" if is_stable else "Sign flips dynamically across OOS periods"
    }
