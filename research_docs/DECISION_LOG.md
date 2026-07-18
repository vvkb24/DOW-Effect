# Decision Log

| Date | Decision | Rationale | Alternatives Considered | Status |
|------|----------|-----------|-------------------------|--------|
| 2026-07-18 | **Pivot to Conditional Alpha Discovery** | Phase 2 falsification of the unconditional Monday effect on NTPC demonstrated that the raw anomaly is a Statistical Artifact ($p = 0.557$, effect size = -0.0004). The research program has been explicitly pivoted to search for interaction-driven edges (e.g., Weekday × Regime). | Terminating the project entirely; however, institutional literature supports regime-dependent anomalies even when unconditional effects fail. | **Active (V2.0)** |
| 2026-07-18 | **Adopt 8-Stage Conditional Funnel** | We need a stricter falsification funnel to handle the exponentially larger feature space of interaction testing, preventing severe data snooping. | Keeping the 6-stage unconditional funnel. | **Active** |
| 2026-07-18 | **Machine Learning for Hypothesis Generation** | Shallow ML models (XGBoost) will be used to rank feature interactions and generate SHAP values to detect non-linear conditional edges, rather than purely as black-box forecasters. | Exhaustive OLS permutations for every possible categorical interaction, which is computationally expensive and prone to multicollinearity. | **Active** |
