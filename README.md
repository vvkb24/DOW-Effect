# Quantitative Research: The "Weekday Effect" in Indian Power Equities

This repository contains an institutional-grade quantitative research pipeline designed to investigate conditional calendar anomalies (specifically the "Weekday Effect") within the Indian power sector. 

The primary objective was to determine whether daily stock returns exhibit predictable bias based on the day of the week, and whether that bias is unlocked by latent market states (volatility, trend, sector rotation).

## Scientific Verdict: Falsified
After a rigorous, multi-phase statistical audit, the hypothesis was **definitively falsified**. The Indian power sector is highly efficient at the daily frequency; any historical profitability observed in simple backtests of the weekday anomaly is a statistical illusion driven by random clustering and data-mining, not structural market inefficiency. 

The research program was formally halted at Phase 2 to **prevent capital deployment** into an overfit, non-generalizable hypothesis. We exhausted the low-frequency feature space (Price, Volume, HMM Regimes, Intermarket Spreads) and identified that moving further would require true institutional flow data (FII/DII stock-specific tick data) which is unavailable publicly.

## Pipeline Architecture
While the hypothesis was false, the outcome is a success. This repository now serves as a robust, reusable engine for hypothesis testing, immunized against common quantitative research pitfalls (lookahead bias, p-hacking, overfitting).

Key architectural features include:
*   **Strict $t-1$ Leakage Prevention:** Every feature is aggressively lagged to ensure models only predict tomorrow's return using today's closing state.
*   **Dynamic Market Regimes (HMM):** Integrates `hmmlearn` to map non-stationary macroeconomic volatility and trend states without lookahead bias.
*   **Exploratory Machine Learning (XGBoost + SHAP):** Uses constrained tree-based models and SHAP value extraction to search for non-linear interactions between market states and calendar days.
*   **Clustered Panel Econometrics:** Uses `linearmodels` (Fixed Effects, Clustered Standard Errors) to estimate true causal effects while controlling for market beta (Nifty 50).
*   **Statistical Discipline:** Implements Bonferroni Family-Wise Error Rate (FWER) corrections and Seasonal-Aware Walk-Forward Out-Of-Sample (OOS) validation.

## Repository Structure
*   `src/doweffect/`: The core quantitative engine.
    *   `features/`: HMM regime mapping, event proximity, returns, and intermarket spread construction.
    *   `ml/`: Exploratory XGBoost discovery and SHAP interaction extraction.
    *   `stats/`: Econometric confirmation testing (Panel OLS) and Walk-Forward OOS splitting.
*   `scripts/`: Execution runners (e.g., `run_hmm_discovery.py`, `run_spreads_discovery.py`).
*   `research_docs/`: Comprehensive institutional memory of the research process, including:
    *   `RESEARCH_LEDGER.md`: The complete log of every hypothesis tested and its statistical verdict.
    *   `EXPERIMENT_REGISTRY.md`: Tracking for all discovery runs.
    *   `RESEARCH_CONTRACT.md`: The governance rules ensuring strict falsification logic.
*   `data/`: Data storage (raw and processed Parquet files are git-ignored to save space; see `data/audit/audit_report.csv` for the dataset timeline spanning 2005-2026).
*   `tests/`: `pytest` suite ensuring architectural integrity and leakage safety.

## Setup and Execution
1.  Initialize a virtual environment: `python -m venv venv`
2.  Activate the environment and install requirements: `pip install -r requirements.txt` (Note: ensure you have `linearmodels`, `hmmlearn`, `xgboost`, `shap`, `pandas`, `yfinance`, and `pytest` installed).
3.  To run the full HMM-based discovery pipeline on the universe: `python scripts/run_hmm_discovery.py`

*Note: The `Docs/` directory is an external user-specific folder and is ignored by Git.*
