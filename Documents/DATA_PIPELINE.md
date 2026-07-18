# Data Engineering Pipeline

This document outlines the data acquisition and feature engineering processes used in this research. The objective of the data pipeline is to construct a rigorous, lookahead-safe dataset for analyzing conditional anomalies in the Indian power sector.

## Data Acquisition

The primary data source for this project is public End-of-Day (EOD) OHLCV (Open, High, Low, Close, Volume) data. 

*   **Source:** Yahoo Finance API (`yfinance`).
*   **Universe:** 14 major Indian power-sector equities (e.g., NTPC, TATAPOWER, POWERGRID, NHPC) and major indices (Nifty 50, Bank Nifty, Nifty Energy).
*   **Timeframe:** Historical data spans from each asset's listing date (some beginning as early as January 2005) up to July 2026. 
*   **Storage:** Raw data is downloaded in bulk, audited for missing values, corporate actions (splits/dividends), and extreme outliers, and then cached locally as Parquet files.

## Feature Engineering and Leakage Prevention

The core constraint of the feature engineering pipeline is strict $t-1$ safety. To ensure that statistical models do not inadvertently train on future information (lookahead bias), every feature used to explain the return on day $t$ is calculated using only information available at the close of day $t-1$.

### How $t-1$ Safety is Achieved
In practice, this is implemented using standard vectorized shifting operations in Pandas. First, all features (e.g., a 20-day moving average of volatility) are calculated using the raw data up to day $t$. Then, the entire feature matrix is explicitly shifted forward by one day (`df.shift(1)`) before it is merged with the target variable (the return on day $t$). 

This mechanical offset guarantees that the target variable $y_t$ (Return on Tuesday) is strictly regressed against the feature vector $X_{t-1}$ (Moving Average at the close of Monday). If a calculation naturally involves a forward lookahead (like HMM smoothing), we restrict the model to only use the "filtered" state (which only sees past data) rather than the "smoothed" state (which sees the entire dataset).

### Engineered Feature Families

1.  **Calendar Features:** One-hot encoded variables representing the day of the week.
2.  **Volatility & Trend:** Rolling standard deviations and moving averages over various lookback windows (e.g., 5-day, 20-day) to capture short-term and medium-term momentum.
3.  **Intermarket Spreads:** Relative strength indicators comparing the performance of the power sector (using Nifty Energy as a proxy) against other sectors, designed to observe sector rotation.
4.  **HMM Regimes:** A Gaussian Hidden Markov Model (`hmmlearn`) trained on historical returns to infer latent, unobservable market states (e.g., "High Volatility" vs. "Low Volatility"). The regime classification is computed on a rolling basis to prevent forward-looking data leakage.

## Pipeline Architecture Diagram

```mermaid
graph TD
    A[Yahoo Finance API] -->|OHLCV Data| B(Raw Data Audit & Caching)
    B -->|Cleaned Parquet| C{Strict t-1 Shifting}
    C --> D[Calendar Dummies]
    C --> E[Rolling Volatility & Trend]
    C --> F[Intermarket Spreads]
    C --> G[HMM Latent State Inference]
    D --> H[Processed Modeling Dataset]
    E --> H
    F --> H
    G --> H
    H -->|Input for Econometrics| I(Panel OLS / XGBoost)
```
