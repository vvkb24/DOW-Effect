# Validation Checklist (Conditional Alpha V2.0)

## Phase 1: Data & Feature Engineering
- [ ] Are all regime flags (Volatility, Trend, Liquidity) computed strictly on `t-1` data to prevent target leakage?
- [ ] Are all rolling window normalizations properly aligned to prevent lookahead bias?
- [ ] Is the feature matrix completely free of NaN/Inf explosions during z-score standardization?

## Phase 2: Conditional Econometrics
- [ ] Do panel regressions explicitly include the main effects of BOTH interaction components alongside the interaction term?
- [ ] Are standard errors double-clustered (Firm and Time) for all panel estimates?
- [ ] Is Westfall-Young or Bonferroni explicitly applied across the *entire* universe of tested interactions to prevent p-hacking?
- [ ] Are SHAP interactions validated against linear OLS interaction coefficients for directionality?

## Phase 3: Out-of-Sample & Validation
- [ ] Does the conditional effect generalize to the holdout set (temporal leakage check)?
- [ ] Does the effect survive random permutation of the regime labels?
- [ ] Is the model overfitting to extremely rare event windows (e.g., exactly 2 specific election days)?
- [ ] Does the interaction remain stable across Bayesian change points?
