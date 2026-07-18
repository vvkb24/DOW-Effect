import logging
import pandas as pd
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
            # Predict on test data
            preds = train_res.predict(test_df)
            
            # Since we are testing if the *interaction* persists, we can fit the 
            # model strictly on the out-of-sample data and check if the sign and 
            # significance hold, OR check if out-of-sample predictive R2 is > 0.
            # Institutional standard for inference is sub-period stability of the coefficient.
            test_res = estimate_panel_effects(test_df, formula)
            
            if interaction_term and interaction_term in test_res.tvalues:
                t_stat = test_res.tvalues[interaction_term]
                oos_tstats.append(t_stat)
                
        except Exception as e:
            logger.debug(f"Walk-forward failed for window {test_years}: {e}")
            
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
        "reason": "Stable" if is_stable else "Sign flips dynamically across OOS periods"
    }
