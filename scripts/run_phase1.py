"""
Phase 1 orchestrator script.
"""
import os
import yaml
import logging
import pandas as pd
from doweffect.data.downloader import Downloader
from doweffect.data.cross_validator import CrossValidator
from doweffect.data.auditor import Auditor
from doweffect.data.validator import Validator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("phase1")

def run():
    logger.info("Starting Phase 1: Data Acquisition and Audit")
    
    with open("config/universe.yaml", "r") as f:
        universe = yaml.safe_load(f)
        
    with open("config/pipeline.yaml", "r") as f:
        pipeline_cfg = yaml.safe_load(f)
        
    start_date = pipeline_cfg["global"]["date_range_start"]
    end_date = pipeline_cfg["global"].get("date_range_end")
    
    downloader = Downloader()
    cross_validator = CrossValidator()
    auditor = Auditor()
    validator = Validator()
    
    audit_results = []
    
    for ticker in universe["tickers"]:
        symbol = ticker["symbol_yf"]
        logger.info(f"Processing {symbol}...")
        
        # Download
        df_adj, df_raw = downloader.download_ticker(symbol, start_date, end_date)
        
        # Cross validate (stub)
        cv_res = cross_validator.compare(df_adj, df_raw, symbol)
        
        # Audit
        audit_res = auditor.audit_ticker(symbol, df_adj, df_raw)
        audit_results.append(audit_res)
        
        # Validate and clean
        if not df_adj.empty:
            file_hash = validator.clean_and_save(symbol, df_adj)
            logger.info(f"Saved cleaned data for {symbol}, hash: {file_hash[:8]}...")
            
    # Process indices
    if "indices" in universe:
        for idx in universe["indices"]:
            symbol = idx["symbol_yf"]
            logger.info(f"Processing index {symbol}...")
            df_adj, df_raw = downloader.download_ticker(symbol, start_date, end_date)
            audit_res = auditor.audit_ticker(symbol, df_adj, df_raw)
            audit_results.append(audit_res)
            if not df_adj.empty:
                validator.clean_and_save(symbol, df_adj)

    # Save audit report
    df_audit = pd.DataFrame(audit_results)
    df_audit.to_csv("data/audit/audit_report.csv", index=False)
    
    # Check minimum coverage (10 years = ~2500 trading days)
    # Filter for primary tickers that have fno=true (just as a heuristic)
    for ticker in universe["tickers"]:
        if ticker.get("fno", False):
            symbol = ticker["symbol_yf"]
            res = df_audit[df_audit["symbol"] == symbol]
            if not res.empty and res.iloc[0]["total_days"] < 2500:
                logger.warning(f"Ticker {symbol} has less than 10 years of data: {res.iloc[0]['total_days']} days.")
                
    # Write complete flag
    with open("data/audit/phase_1_complete.flag", "w") as f:
        f.write("Phase 1 completed successfully.\n")
        
    logger.info("Phase 1 completed. Wrote phase_1_complete.flag")

if __name__ == "__main__":
    run()
