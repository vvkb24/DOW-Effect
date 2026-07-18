import os
import glob
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to python path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from doweffect.features.returns import compute_raw_returns
from doweffect.features.regimes import compute_regimes
from doweffect.features.events import compute_events
from doweffect.features.interactions import generate_interaction_matrix
from doweffect.ml.discovery import train_exploratory_model
from doweffect.ml.interpretability import extract_shap_interactions
from doweffect.stats.validation import walk_forward_validation
from doweffect.stats.panel import estimate_panel_effects

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("conditional_discovery")

def main():
    logger.info("Starting Conditional Alpha Discovery Engine (V2.0)")
    
    processed_dir = "data/processed/"
    target_ticker = "NTPC.NS"
    target_path = os.path.join(processed_dir, f"{target_ticker}.parquet")
    
    if not os.path.exists(target_path):
        logger.error(f"Target ticker {target_ticker} not found in processed.")
        return
        
    df = pd.read_parquet(target_path)
    df = compute_raw_returns(df)
    
    # 1. Feature Layer (t-1 safe)
    df = compute_regimes(df, vol_window=21, trend_window=50)
    df = compute_events(df)
    
    # 2. Market Neutrality
    nifty_path = os.path.join(processed_dir, "^NSEI.parquet")
    if os.path.exists(nifty_path):
        nifty_df = compute_raw_returns(pd.read_parquet(nifty_path))
        df = df.join(nifty_df[['LogReturn']], rsuffix='_Market').dropna(subset=['LogReturn', 'LogReturn_Market'])
        df['ResidualReturn'] = df['LogReturn'] - df['LogReturn_Market']
    else:
        df['ResidualReturn'] = df['LogReturn']
        
    # 3. Interactions
    df, interaction_cols = generate_interaction_matrix(df, min_support=30)
    
    # 4. ML Discovery Layer
    feature_cols = [c for c in df.columns if c.startswith('regime_') or c.startswith('is_') or c.startswith('day_')]
    model, X = train_exploratory_model(df, feature_cols, target_col='ResidualReturn')
    
    if model:
        # 5. Interpretability
        shap_candidates = extract_shap_interactions(model, X)
        logger.info(f"Top 3 SHAP Candidates:\n{shap_candidates.head(3)}")
        
        top_candidate = shap_candidates.iloc[0]
        weekday = top_candidate['Weekday']
        condition = top_candidate['Condition']
        
        logger.info(f"ML suggests testing: {weekday} interacting with {condition}")
        
        # 6. Statistical Confirmation
        formula = f"ResidualReturn ~ {weekday} * {condition}"
        try:
            res = estimate_panel_effects(df, formula)
            interaction_term = f"{weekday}:{condition}"
            if interaction_term in res.tvalues:
                t_stat = res.tvalues[interaction_term]
                p_val = res.pvalues[interaction_term]
                logger.info(f"Confirmation OLS -> {interaction_term} t-stat: {t_stat:.2f}, p-value: {p_val:.4f}")
                
                # 7. Walk-Forward OOS
                oos_res = walk_forward_validation(df, formula)
                logger.info(f"Walk-Forward Validation: {oos_res}")
                
        except Exception as e:
            logger.error(f"Failed to run confirmation OLS: {e}")
            
    logger.info("Conditional Alpha Discovery Pipeline Complete.")

if __name__ == "__main__":
    main()
