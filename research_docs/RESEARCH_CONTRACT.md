# Research Contract (Version 2.0 - Conditional Alpha Discovery)

## 1. Research Objective
The project seeks to identify whether Indian power-sector equities exhibit conditional alpha that depends on the joint state of weekday, regime, volatility, liquidity, trend, sector rotation, and event context, rather than a standalone unconditional weekday anomaly.

**Primary Question:** Under what specific combinations of weekday, market regime, volatility, liquidity, trend, and event conditions do Indian power-sector equities exhibit statistically and economically significant conditional return behavior?
**Secondary Question:** Do these interaction-driven patterns survive rigorous out-of-sample multiple-testing corrections and economic friction modeling?
**Exploratory Question:** Can machine learning models acting as hypothesis generation engines (XGBoost, LightGBM) identify stable, non-linear interactions without overfitting?

## 2. Research Philosophy
1. A simple unconditional weekday effect may fail while a conditional effect still exists.
2. A discovered interaction is only valuable if it survives multiple-testing correction, out-of-sample validation, and economic friction.
3. Statistical significance alone is not enough; economic significance alone is not enough.
4. A pattern is not real unless it persists across time, regimes, and data splits.
5. The project must actively attempt to falsify every candidate conditional alpha.
6. Do not force the data to confirm a hypothesis. Let the data determine structure.

## 3. Methodological Expansion
### 3.1 Regime Layer
Testing must be conditioned on explicitly defined market regimes: bull/bear, high/low volatility, risk-on/risk-off, and specific event proximity (expiry week, pre-holiday).

### 3.2 Interaction Layer
The statistical engine will now test interaction terms (e.g., `Monday × High Volatility`, `Friday × Post-Holiday`) using panel regressions with clustered standard errors.

### 3.3 Feature Space
Expanding features to include: lagged returns, overnight gaps, realized volatility (ATR), trend distances, sector relative strength, liquidity proxies, volume z-scores, breadth, and event flags.

## 4. Decision Framework (Conditional Falsification Funnel)
1. **Stage 1 (Unconditional Base):** Does any unconditional weekday effect exist? (If no, continue to conditional testing).
2. **Stage 2 (Interaction Detection):** Does the weekday effect appear under specific regimes or interaction states?
3. **Stage 3 (Multiple-Testing):** Does the interaction survive Family-Wise Error Rate (FWER) / Westfall-Young correction? (If no, reject as data mining).
4. **Stage 4 (OOS Validation):** Does the interaction survive out-of-sample and walk-forward testing? (If no, reject as unstable).
5. **Stage 5 (Economic Viability):** Is the interaction economically meaningful before costs?
6. **Stage 6 (Deployability):** Can the interaction be converted into a robust strategy after realistic transaction costs and slippage?
7. **Stage 7 (Structural Stability):** Does the strategy survive subperiod stress tests and structural breaks?
8. **Stage 8 (Classification):** Classify strictly into one of: No Evidence, Unconditional Effect Rejected, Conditional Effect Detected, Conditional Effect Not Robust, Economically Significant but Not Deployable, Robust Conditional Alpha, Deployable Trading Edge.

## 5. Methodology Amendment Policy
Methodology changes are recorded strictly in the DECISION_LOG.md. The unconditional methodology was formally superseded by Version 2.0 due to Stage 2 statistical failure of the unconditional Monday effect.
