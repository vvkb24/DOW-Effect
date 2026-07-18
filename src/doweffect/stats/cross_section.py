import pandas as pd
import statsmodels.api as sm

def petersen_clustered_ols(y: pd.Series, X: pd.DataFrame, firm_ids: pd.Series, time_ids: pd.Series):
    """
    Fits an OLS regression and computes Petersen (2009) two-way 
    cluster-robust standard errors (clustered by Firm and Time).
    
    This effectively corrects standard errors when observations are correlated 
    both cross-sectionally (across stocks on the same day) and 
    time-series-wise (within the same stock over time).
    """
    # Combine to ensure alignment
    df = pd.concat([y.rename("Y"), X, firm_ids.rename("Firm"), time_ids.rename("Time")], axis=1).dropna()
    
    Y = df["Y"]
    X_aligned = df.drop(columns=["Y", "Firm", "Time"])
    
    groups = df[["Firm", "Time"]]
    
    model = sm.OLS(Y, X_aligned)
    # cov_type='cluster' with groups spanning two dimensions implements Petersen (2009) multi-way clustering
    result = model.fit(cov_type='cluster', cov_kwds={'groups': groups})
    
    return result
