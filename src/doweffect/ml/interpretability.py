import logging
import pandas as pd
import numpy as np
import shap
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)

def extract_shap_interactions(model: XGBRegressor, X: pd.DataFrame) -> pd.DataFrame:
    """
    Computes SHAP interaction values to rank the importance of conditional weekday dependencies.
    """
    # TreeExplainer is exact for XGBoost
    explainer = shap.TreeExplainer(model)
    
    # Compute interaction values (returns shape: [n_samples, n_features, n_features])
    # Note: For large datasets, this can be extremely slow. We sample if N > 5000.
    if len(X) > 5000:
        X_sample = X.sample(5000, random_state=42)
    else:
        X_sample = X
        
    shap_interaction_values = explainer.shap_interaction_values(X_sample)
    
    # Average absolute interaction value over all samples
    mean_abs_interactions = np.abs(shap_interaction_values).mean(axis=0)
    
    # Convert to DataFrame
    interaction_df = pd.DataFrame(
        mean_abs_interactions, 
        index=X.columns, 
        columns=X.columns
    )
    
    # Extract only interactions involving weekday
    day_cols = [c for c in X.columns if c.startswith('day_')]
    
    candidates = []
    for day in day_cols:
        for feat in X.columns:
            if feat != day and not feat.startswith('day_'):
                score = interaction_df.loc[day, feat]
                candidates.append({
                    'Weekday': day,
                    'Condition': feat,
                    'SHAP_Interaction_Score': score
                })
                
    results = pd.DataFrame(candidates).sort_values(by='SHAP_Interaction_Score', ascending=False)
    logger.info("SHAP Interaction extraction complete.")
    return results
