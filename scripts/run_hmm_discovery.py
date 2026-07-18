import os
import yaml
import logging
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from doweffect.features.returns import compute_raw_returns
from doweffect.features.hmm_regimes import compute_hmm_regimes
from doweffect.features.events import compute_events
from doweffect.features.interactions import generate_interaction_matrix
from doweffect.ml.discovery import train_exploratory_model
from doweffect.ml.interpretability import extract_shap_interactions
from doweffect.stats.validation import walk_forward_validation
from doweffect.stats.panel import estimate_panel_effects

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("hmm_discovery")

def load_universe():
    with open("config/universe.yaml", 'r') as f:
        config = yaml.safe_load(f)
    return [t['symbol_yf'] for t in config['tickers']]

def main():
    logger.info("Starting Universe HMM Conditional Alpha Discovery (Phase 2.2)")
    
    universe = load_universe()
    processed_dir = "data/processed/"
    nifty_path = os.path.join(processed_dir, "^NSEI.parquet")
    
    if os.path.exists(nifty_path):
        nifty_df = compute_raw_returns(pd.read_parquet(nifty_path))
    else:
        logger.error("Nifty data missing, cannot compute residuals.")
        return

    all_stock_dfs = []
    verdicts = []
    
    for ticker in universe:
        logger.info(f"--- Processing {ticker} ---")
        target_path = os.path.join(processed_dir, f"{ticker}.parquet")
        if not os.path.exists(target_path):
            continue
            
        df = pd.read_parquet(target_path)
        df = compute_raw_returns(df)
        
        # 1. Feature Layer (HMM + Events ONLY, no static regimes)
        df = compute_hmm_regimes(df, window_size=252, step=21)
        df = compute_events(df)
        
        # Market Neutrality
        df = df.join(nifty_df[['LogReturn']], rsuffix='_Market').dropna(subset=['LogReturn', 'LogReturn_Market'])
        df['ResidualReturn'] = df['LogReturn'] - df['LogReturn_Market']
        
        # 2. Interactions (Max Depth is handled inside train_exploratory_model)
        df, interaction_cols = generate_interaction_matrix(df, min_support=30)
        df['Firm'] = ticker
        
        all_stock_dfs.append(df)
        
        # 3. ML Discovery
        feature_cols = [c for c in df.columns if c.startswith('regime_hmm') or c.startswith('is_') or c.startswith('day_')]
        if len(feature_cols) == 0:
            continue
            
        model, X = train_exploratory_model(df, feature_cols, target_col='ResidualReturn')
        
        stock_verdict = "No Candidates"
        if model:
            shap_candidates = extract_shap_interactions(model, X)
            if not shap_candidates.empty:
                top_candidate = shap_candidates.iloc[0]
                weekday = top_candidate['Weekday']
                condition = top_candidate['Condition']
                interaction_term = f"{weekday}:{condition}"
                formula = f"ResidualReturn ~ {weekday} * {condition}"
                
                try:
                    res = estimate_panel_effects(df, formula)
                    if interaction_term in res.tvalues:
                        p_val = res.pvalues[interaction_term]
                        # FWER correction (Bonferroni approximation for N interactions)
                        fwer_p = min(p_val * len(interaction_cols), 1.0)
                        
                        if fwer_p < 0.05:
                            oos_res = walk_forward_validation(df, formula, seasonal_aware=True)
                            if oos_res['stable']:
                                stock_verdict = f"Robust HMM Alpha ({interaction_term})"
                            else:
                                stock_verdict = f"Unstable HMM OOS ({interaction_term})"
                        else:
                            stock_verdict = f"Rejected FWER ({interaction_term}, p={fwer_p:.3f})"
                except Exception as e:
                    stock_verdict = "OLS Failed"
                
        verdicts.append({"Firm": ticker, "Verdict": stock_verdict})

    # Output Stock-by-stock
    verdict_df = pd.DataFrame(verdicts)
    logger.info(f"\n{verdict_df.to_string()}")
    
    # 4. Pooled Universe Aggregation
    logger.info("--- Pooled Universe Analysis ---")
    pooled_df = pd.concat(all_stock_dfs)
    
    feature_cols = [c for c in pooled_df.columns if c.startswith('regime_hmm') or c.startswith('is_') or c.startswith('day_')]
    model, X = train_exploratory_model(pooled_df, feature_cols, target_col='ResidualReturn')
    
    if model:
        shap_candidates = extract_shap_interactions(model, X)
        logger.info(f"Top 3 Pooled SHAP Candidates:\n{shap_candidates.head(3)}")
        top_candidate = shap_candidates.iloc[0]
        weekday = top_candidate['Weekday']
        condition = top_candidate['Condition']
        interaction_term = f"{weekday}:{condition}"
        formula = f"ResidualReturn ~ C(Firm) + {weekday} * {condition}"
        
        logger.info(f"Pooled Confirmation Testing: {formula}")
        try:
            res = estimate_panel_effects(pooled_df, formula)
            if interaction_term in res.tvalues:
                t_stat = res.tvalues[interaction_term]
                p_val = res.pvalues[interaction_term]
                fwer_p = min(p_val * len(feature_cols), 1.0)
                logger.info(f"Pooled {interaction_term} | Adj p-val: {fwer_p:.4f}")
                
                oos_res = walk_forward_validation(pooled_df, formula, seasonal_aware=True)
                logger.info(f"Pooled OOS Validation: {oos_res['reason']}")
                
                if fwer_p < 0.05 and oos_res['stable']:
                    logger.info("VERDICT: Sector-Wide Robust HMM Alpha Detected.")
                else:
                    logger.info("VERDICT: Sector-Wide HMM Conditional Alpha Rejected.")
        except Exception as e:
            logger.error(f"Pooled OLS failed: {e}")

if __name__ == "__main__":
    main()
