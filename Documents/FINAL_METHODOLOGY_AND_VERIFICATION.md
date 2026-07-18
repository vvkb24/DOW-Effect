# Final Methodology and Verification Report

This document serves as the permanent methodological reference for the investigation into conditional calendar anomalies within the Indian power sector. It is designed as an institutional reference to allow new quantitative researchers to understand, reproduce, audit, and extend the project.

---

## 1. Project Overview

### Original Hypothesis
The research investigated the "Weekday Effect" within the Indian power sector. The core hypothesis tested whether daily stock returns exhibit a predictable, statistically significant bias based on the day of the week, and whether this bias is unlocked by latent market states (e.g., volatility regimes, trend states, sector rotation).

### The Value of Falsification
The hypothesis was structurally falsified. Falsification is a highly valuable outcome in quantitative research. The objective of an institutional research pipeline is not to force an anomaly to be profitable, but to actively attempt to break it. Proving that an anomaly does not exist prevents capital deployment into overfit, non-generalizable strategies. 

### Research Philosophy
The project adhered to a strict research constitution:
* **Evidence > Intuition**: All conclusions require strict statistical evidence.
* **Leakage Prevention**: All features must strictly observe a $t-1$ constraint.
* **Separation of ML and Econometrics**: Machine Learning is strictly isolated for discovery; Econometrics is used exclusively for confirmation.
* **Aggressive Falsification**: The burden of proof is on the anomaly; it must survive adversarial walk-forward validation and multiple-testing corrections.

---

## 2. Complete Research Timeline

### Phase 1: Unconditional Weekday Testing
* **Objective:** Test for a simple, unconditional weekday effect.
* **Implementation:** Panel OLS regressing residual returns against day-of-week dummy variables.
* **Findings:** No statistically significant unconditional effect.
* **Action:** Continued to Phase 2 to test conditional anomalies.

### Phase 2.1: Conditional Price/Volume Features
* **Objective:** Test if simple momentum or liquidity states unlock a weekday effect.
* **Implementation:** Rolling standard deviations, moving averages, and volume proxies interacted with weekday dummies.
* **Findings:** Initial in-sample significance was found, but it collapsed under FWER corrections and OOS validation.
* **Action:** Continued to Phase 2.2 for advanced state mapping.

### Phase 2.2: HMM Dynamic Regimes
* **Objective:** Identify latent macroeconomic states (high vs. low volatility/trend) that might condition the anomaly.
* **Implementation:** A Gaussian Hidden Markov Model (`hmmlearn`) was fitted on historical returns on a rolling basis to infer latent states without lookahead bias.
* **Findings:** HMM regimes successfully segregated high and low volatility periods, but interaction with weekday effects failed walk-forward validation.
* **Action:** Continued to Phase 2.3.

### Phase 2.3: Intermarket Spreads
* **Objective:** Test if sector rotation (Power vs. Metal, Power vs. Nifty50) drives conditional weekday effects.
* **Implementation:** Engineered relative strength spreads shifted to $t-1$.
* **Findings:** Spreads provided no predictive power over weekday anomalies out-of-sample.
* **Action:** Reached the limits of public low-frequency data.

### Phase 2.4: Institutional Flow Audit (Project Termination)
* **Objective:** Determine whether to proceed with high-frequency institutional flow data.
* **Implementation:** Evaluated data availability. High-quality tick and flow data was unavailable.
* **Findings/Action:** Terminated. Rather than data-mining derived OHLCV proxies (which risks overfitting), the research reached its predefined stopping criterion. The negative finding was accepted as a scientific reality.

---

## 3. Statistical Architecture

### Panel Regression & Firm Fixed Effects
The primary inference engine is a clustered Panel OLS (`linearmodels` / `statsmodels`). Firm fixed effects correctly absorb static baseline outperformance of individual assets, isolating the time-varying weekday effects.

### Multiple Testing (FWER & Bonferroni)
Testing dozens of feature combinations exponentially increases the risk of false positives. The pipeline implements Family-Wise Error Rate (FWER) controls via Bonferroni corrections to mathematically bound the risk of p-hacking.

### Walk-Forward Validation
To ensure structural persistence, models are validated dynamically over expanding temporal windows, eliminating lookahead bias and survivorship bias inherent in static hold-out sets.

### HMM & ML Separation (XGBoost + SHAP)
* **ML (Discovery):** XGBoost is used exclusively to navigate the combinatorial feature space heuristically, utilizing SHAP to propose non-linear interactions.
* **Econometrics (Confirmation):** ML predictions are discarded. Only the *symbolic interactions* (e.g., `day_0 * regime_high_vol`) are passed to the Panel OLS engine to estimate true causal effects. ML never performs confirmation.

---

## 4. Leakage Prevention

The pipeline acts defensively against all forms of temporal data leakage:
* **t-1 Safety:** Every feature is explicitly shifted forward by one day (`df.shift(1)`). To predict $y_t$, the model strictly uses $X_{t-1}$.
* **HMM Construction:** The HMM is trained up to $t-2$ and predicts $t-1$, which is then aligned with the target at $t$. It relies solely on the "filtered" state, explicitly avoiding the forward-looking "smoothed" state.
* **Rolling Windows:** Standard rolling metrics (e.g., 20-day volatility) compute up to $t$, and are then mechanically shifted to ensure $t$ does not contaminate its own prediction.
* **Walk-Forward Alignment:** Walk-forward bounds are strictly respected; no test window ever leaks into the training window's normalization or estimation logic.

---

## 5. Validation Pipeline

The validation engine requires an anomaly to survive two distinct adversarial tests:
1. **Structural Persistence:** A coefficient must maintain its sign and statistical significance dynamically across out-of-sample periods. If it flips signs, the anomaly is unstable and untradeable.
2. **Predictive Validation:** The model trained on historical data must successfully predict the magnitude and direction of the target variable in unseen data. 
Both are required because an anomaly might temporarily exist (structural) but be too noisy to harvest effectively (predictive).

---

## 6. Predictive OOS Extension

### Rationale
The engine was extended to explicitly separate structural inference from execution reality. A coefficient can be stable but still produce negative predictive value if the variance dominates the signal.

### Mechanics
1. $\beta_{train}$ is extracted by fitting the Panel OLS solely on the expanding training window.
2. The design matrix $X_{test}$ is constructed for the unseen test window.
3. Predictions are generated strictly via $\hat{y}_{test} = X_{test} \beta_{train}$.
4. Metrics computed: **RMSE**, **MAE**, **Bias** (Mean Error), and **Out-of-Sample $R^2$**.
5. *Coefficient stability* is computed subsequently by refitting the model on the test block to check structural presence.

### Mathematical Distinction
Predictive validation evaluates $\epsilon = y_{test} - X_{test} \hat{\beta}_{train}$ (could we trade this?). Coefficient stability evaluates $\hat{\beta}_{test}$ (did the anomaly exist in this sub-period?). 

---

## 7. Bug History

### 1. Statsmodels NaN Alignment Bug
* **Description:** During the implementation of the Predictive OOS extension, it was discovered that `statsmodels` silently drops $NA$ rows when running its `.predict()` method if columns referenced in the patsy formula contain missing values.
* **Root Cause:** A list comprehension successfully dropped $NA$s for columns present in `test_df.columns`, but ignored patsy transformations like `C(DayOfWeek)`, causing them to remain in the dataset and be silently dropped by statsmodels.
* **Impact:** The resulting `preds` Series was shorter than `y_test`. When computing $R^2$, Pandas aligned the indices for subtraction, resulting in $NA$s, but the Total Sum of Squares (SST) was computed over the entire `y_test` array. This asymmetrical calculation invalidated the $R^2$ math.
* **Fix:** Explicitly forced `y_test` to align to the valid prediction index via `y_test = valid_test_df.loc[preds.dropna().index, target_col]`.
* **Validation:** Re-running the walk-forward engine correctly yielded a mathematically symmetrical and accurate negative $R^2$ for the null hypothesis.

### 2. Statsmodels Singleton Cluster ZeroDivisionError
* **Description:** While verifying the validation engine using a mocked synthetic dataset, `statsmodels.stats.sandwich_covariance` raised a `ZeroDivisionError: float division by zero`.
* **Root Cause:** The synthetic data mock assigned all rows to a single firm (`"Firm": "TEST_FIRM"`). Statsmodels cluster-robust covariance matrix uses a degrees-of-freedom correction involving `n_groups / (n_groups - 1.0)`. For a single firm, `1 - 1 = 0`.
* **Impact:** The test script crashed.
* **Fix:** Updated the synthetic mock to randomly distribute rows across `["FIRM_A", "FIRM_B", "FIRM_C"]`, ensuring $N_{clusters} > 1$. Furthermore, the data type was cast to integers to prevent a NumPy object-array view error within `grouputils`.
* **Validation:** Walk-forward verification executed flawlessly.

---

## 8. Scientific Conclusions

**Under the tested feature space, timeframe, assets and validation framework, no robust conditional weekday alpha was found.**

* **Proven:** We have mathematically proven that relying on historical price, volume, structural volatility (HMM), and intermarket spreads at a daily frequency is insufficient to extract a reliable weekday edge in the Indian power sector.
* **Not Proven:** We did *not* prove that the market is perfectly efficient in an absolute sense. 
* **Limitations:** The conclusion is bounded by the daily EOD resolution of the data and the specific feature families tested. Anomaly exploitation may require fundamentally different datasets (tick data, institutional flow).

---

## 9. Repository Architecture

* `src/doweffect/features/`: Contains the strictly $t-1$ lagged feature generators, including the rolling metrics, intermarket spreads, and the safe `hmm_regimes.py` engine.
* `src/doweffect/ml/`: Houses the XGBoost and SHAP integration for heuristic interaction discovery, quarantined from final inference.
* `src/doweffect/stats/`: Contains the core econometric confirmation tools (`panel.py`) and the rigorous walk-forward validation framework (`validation.py`).
* `scripts/`: Execution runners that orchestrate the pipeline from data loading to metric reporting.
* `tests/`: Pytest suite explicitly verifying architectural integrity, leakage safety, and mathematical correctness.

---

## 10. Lessons Learned

* **Research:** Falsification is a success state. Data mining derived proxies from a barren dataset is the single greatest risk to institutional capital.
* **Statistical:** The separation of ML discovery from econometric confirmation provides the perfect balance of heuristic navigation and causal validity.
* **Engineering:** Validating predictions requires obsessive attention to Pandas index alignment; silent $NA$ dropping in statistical libraries can quietly destroy downstream metrics.
* **Software:** A highly modular pipeline is the only way to swap between unconditional testing and complex HMM state inferences without introducing temporal leakage.

---

## 11. Future Research

Adding more daily OHLCV-derived features presents an extreme risk of p-hacking and feature fishing. Future alpha research should pivot to qualitatively different dimensions:
1. **Intraday Microstructure:** 1-minute, 5-minute, or Level 2 order-book data.
2. **Options & Volatility Risk Premia:** Analyzing implied vs. realized volatility, Greeks, and skew dynamics.
3. **Cross-Sectional Statistical Arbitrage:** Cointegration, pairs trading, and residual mean reversion.
4. **Alternative Data:** Institutional flow, macroeconomic surprises, or parsed news sentiment.

---

## 12. Final Verification

* **Executive Verdict:** PASS
* **Leakage Verification:** PASS (All features strictly bound to $t-1$)
* **Mathematical Verification:** PASS (OOS RMSE, MAE, Bias, and $R^2$ compute correctly with proper index alignment)
* **Predictive Verification:** PASS (OOS predictive evaluation correctly isolated from train parameters)
* **Scientific Conclusion:** CONFIRMED

The negative scientific conclusion is structurally sound, mathematically defended, and immune to lookahead or survivorship bias.

---

## 13. Appendix

* **Glossary:** 
  * $t$: Execution Day.
  * $t-1$: Information State Day.
  * **FWER:** Family-Wise Error Rate (the probability of making at least one Type I error across all tests).
* **Core Equations:**
  * Strict Predictive OOS: $\hat{y}_{test} = X_{test} \beta_{train}$
* **Governance Documents:** `AGENTS.md`, `README.md`, `docs/DATA_PIPELINE.md`.
