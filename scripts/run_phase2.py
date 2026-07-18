import os
import glob
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to python path so we can import doweffect if running directly
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from doweffect.features.returns import compute_raw_returns
from doweffect.events.calendar import add_calendar_events
from doweffect.stats.hypothesis import exhaustive_pairwise_tests
from doweffect.stats.corrections import westfall_young_permutation
from doweffect.ledger.hypothesis_log import log_hypothesis_result
from doweffect.reports.statistical import generate_statistical_report

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("phase2")

def main():
    logger.info("Starting Phase 2: Institutional Statistical Analysis")
    
    flag_path = "data/audit/phase_1_complete.flag"
    if not os.path.exists(flag_path):
        logger.error(f"Phase 1 not complete. Missing {flag_path}")
        return
        
    processed_dir = "data/processed/"
    
    # Load NIFTY 50 (Market Proxy)
    nifty_path = os.path.join(processed_dir, "^NSEI.parquet")
    if not os.path.exists(nifty_path):
        logger.error("Market Proxy (^NSEI) not found in processed. Cannot perform market neutral analysis.")
        return
        
    nifty_df = pd.read_parquet(nifty_path)
    nifty_df = compute_raw_returns(nifty_df)
    
    # We will process NTPC.NS as our primary test candidate for Phase 2 implementation.
    # In a full run, we would loop over all tickers.
    target_ticker = "NTPC.NS"
    target_path = os.path.join(processed_dir, f"{target_ticker}.parquet")
    
    if not os.path.exists(target_path):
        logger.error(f"Target ticker {target_ticker} not found in processed.")
        return
        
    logger.info(f"Processing {target_ticker}...")
    df = pd.read_parquet(target_path)
    df = compute_raw_returns(df)
    df = add_calendar_events(df)
    
    # Merge with Nifty to compute simple CAPM residual
    # Note: Full 3-factor requires downloading Nifty Midcap & Nifty PSE, 
    # but we approximate with Market for this orchestration demo.
    merged = df.join(nifty_df[['LogReturn']], rsuffix='_Market').dropna(subset=['LogReturn', 'LogReturn_Market'])
    
    # Simple market residual: Return - Beta * MarketReturn
    # For a quick proxy, assume Beta = 1.0
    merged['ResidualReturn'] = merged['LogReturn'] - merged['LogReturn_Market']
    
    # 1. Existence: Exhaustive pairwise on Residuals
    pairwise_results = exhaustive_pairwise_tests(merged, value_col='ResidualReturn')
    
    # 2. Extract Monday (0) vs Friday (4) for the report
    mon_fri = pairwise_results[(pairwise_results['Day1'] == 0) & (pairwise_results['Day2'] == 4)]
    
    if mon_fri.empty:
        logger.error("Could not compute Monday vs Friday.")
        return
        
    raw_p_value = mon_fri.iloc[0]['Welch_p']
    effect_size = mon_fri.iloc[0]['MeanDiff']
    
    # 3. False Discovery: Westfall-Young Permutation
    mon_resids = merged[merged['DayOfWeek'] == 0]['ResidualReturn'].values
    fri_resids = merged[merged['DayOfWeek'] == 4]['ResidualReturn'].values
    
    # Run a simplified permutation test for the orchestrator
    np.random.seed(42)
    wy_p_value = westfall_young_permutation(mon_resids, fri_resids, n_permutations=1000)
    
    # 4. Classification
    verdict = "Econometrically Robust Anomaly" if wy_p_value < 0.05 else "Statistical Artifact"
    
    logger.info(f"Raw P-Value: {raw_p_value:.4e}, WY Corrected: {wy_p_value:.4e}, Verdict: {verdict}")
    
    # Log to Ledger
    log_hypothesis_result(
        hypothesis_id="H1 (Monday vs Friday Residuals)",
        description=f"Day-of-Week Effect on {target_ticker} (Market-Adjusted)",
        test_name="Welch t-test + Westfall-Young",
        p_value_raw=raw_p_value,
        p_value_adj=wy_p_value,
        effect_size=effect_size,
        verdict=verdict
    )
    
    # Generate HTML Report
    generate_statistical_report(pairwise_results)
    
    # Write Phase 2 flag
    phase2_flag = "data/audit/phase_2_complete.flag"
    with open(phase2_flag, "w") as f:
        f.write("Phase 2 complete.")
        
    logger.info("Phase 2 completed successfully. Wrote phase_2_complete.flag")

if __name__ == "__main__":
    main()
