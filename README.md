# The "Weekday Effect" in Indian Power Equities

This repository investigates conditional calendar anomalies (specifically the "Weekday Effect") within the Indian power sector. 

The primary objective was to determine whether daily stock returns exhibit predictable bias based on the day of the week, and whether that bias is influenced by latent market states (volatility, trend, sector rotation).

## Scientific Verdict: Hypothesis Falsified

**Within the feature families, assets, timeframe, and statistical framework investigated, we found no reproducible conditional alpha.**

Despite progressively increasing model complexity—from unconditional weekday tests through conditional regimes, HMM state inference, and intermarket spread features—we found no evidence of economically meaningful, out-of-sample stable weekday-related alpha in Indian power-sector equities using publicly available daily data. This suggests that any exploitable inefficiency, if present, either exists outside the examined feature space, at a different temporal resolution, or requires information unavailable in public daily datasets.

## Project Termination: A Methodological Success

The decision was made to halt the research program after exhausting available daily price and volume features. This was a deliberate choice to preserve scientific integrity. 

When high-quality institutional flow data was deemed unavailable, the project faced a methodological crossroads: either invent derived proxies from existing low-frequency data, or terminate the search. 

Choosing to invent proxies from data already proven to lack predictive power introduces severe data mining risks. Instead, the project was concluded upon reaching a predefined stopping criterion. Maintaining strict methodological standards—such as avoiding proxy mining, retaining strict out-of-sample validation, and applying stringent multiple-testing corrections—ensures that the finding of "no robust edge" remains a highly valuable, scientifically defensible result. This discipline successfully prevents capital deployment into overfit, non-generalizable hypotheses.

## The True Asset: The Research Engine

The most important outcome of this project was not the anomaly itself, but the creation of a rigorous testing **engine**. This repository serves as a reusable framework designed to mitigate common research pitfalls.

```mermaid
graph TD
    A[Raw OHLCV Data] --> B(Strict t-1 Lagging)
    B --> C[Feature Engineering]
    C --> D{Hidden Markov Models}
    D -->|Volatility Regimes| E(Exploratory ML)
    D -->|Trend Regimes| E
    E -->|XGBoost + SHAP| F[Candidate Interactions]
    F --> G{Econometric Confirmation}
    G -->|Clustered Panel OLS| H[Walk-Forward Validation]
    H -->|FWER Corrected| I((Statistical Falsification))
```

Key architectural features built into the engine include:
*   **Leakage-Safe Pipeline:** Strict $t-1$ lagging to ensure models only predict tomorrow's return using today's closing state.
*   **Dynamic Market Regimes (HMM):** Integrates `hmmlearn` to map non-stationary macroeconomic states without lookahead bias.
*   **Exploratory Discovery Layer:** Uses constrained XGBoost models and SHAP value extraction to search for non-linear interactions.
*   **Panel Econometrics:** Uses `linearmodels` (Fixed Effects, Clustered Standard Errors) to estimate true causal effects while controlling for market beta.
*   **Statistical Discipline:** Implements Bonferroni Family-Wise Error Rate (FWER) corrections and Walk-Forward Out-Of-Sample validation.
*   **Reproducible Workflow:** Strict research governance, experiment registries, and hypothesis ledgers.

## Repository Structure
*   `src/doweffect/`: Core modules.
    *   `features/`: HMM regime mapping, event proximity, returns, and intermarket spread construction.
    *   `ml/`: Exploratory XGBoost discovery and SHAP interaction extraction.
    *   `stats/`: Econometric confirmation testing (Panel OLS) and Walk-Forward OOS splitting.
*   `scripts/`: Execution runners (e.g., `run_hmm_discovery.py`, `run_spreads_discovery.py`).
*   `data/`: Data storage (raw and processed Parquet files are git-ignored; see `data/audit/audit_report.csv` for the dataset timeline spanning 2005-2026).
*   `tests/`: `pytest` suite for integrity and leakage checks.

## Future Research Directions
Having exhausted unconditional weekday, volatility, trend, liquidity, HMM, and intermarket spread features on daily OHLCV data, future meaningful directions involve qualitatively different dimensions:
1. **Intraday Microstructure:** (1-minute, 5-minute, order-book data).
2. **Options & Volatility Risk Premia:** (Implied vs. realized volatility, Greeks, skew).
3. **Cross-Sectional Statistical Arbitrage:** (Cointegration, pairs trading, residual mean reversion).

## Setup and Execution
1.  Initialize a virtual environment: `python -m venv venv`
2.  Activate the environment and install requirements: `pip install -r requirements.txt`
3.  To run the full HMM-based discovery pipeline on the universe: `python scripts/run_hmm_discovery.py`
