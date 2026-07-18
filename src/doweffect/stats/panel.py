import pandas as pd
import statsmodels.formula.api as smf

def estimate_panel_effects(df: pd.DataFrame, formula: str) -> object:
    """
    Estimates weekday coefficients using a fixed-effects style panel regression.
    Since proper panel models require linearmodels which is not in our requirements,
    we use statsmodels OLS with explicit dummy variables for fixed effects.
    
    Formula example: 'Return ~ C(DayOfWeek) + C(is_t1_settlement) + Amihud_Illiquidity'
    """
    # Clean formula string to extract actual column names
    import re
    cols = re.sub(r'[~+*:]', ' ', formula).split()
    # Filter out patsy functions like C(...) if any, though we should just keep valid df columns
    cols = [c for c in cols if c in df.columns]
    df = df.dropna(subset=cols)    
    if df.empty:
        raise ValueError("DataFrame is empty after dropping NAs for the panel formula.")
        
    model = smf.ols(formula=formula, data=df)
    # Fit with Petersen clustering (Time and Firm).
    # Assumes 'Time' and 'Firm' are columns in the dataframe.
    if 'Time' in df.columns and 'Firm' in df.columns:
        result = model.fit(cov_type='cluster', cov_kwds={'groups': df[['Firm', 'Time']]})
    else:
        # Fallback to White's HC3 robust standard errors if cluster dims not provided
        result = model.fit(cov_type='HC3')
        
    return result
