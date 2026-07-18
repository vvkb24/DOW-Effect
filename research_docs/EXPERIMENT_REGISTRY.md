# Experiment Registry (Conditional Alpha)

This registry strictly tracks every interaction experiment to prevent undocumented data snooping.

| Exp ID | Date | Feature Set / Interaction | Regime Definition | Model / Test | Significance Correction | OOS Results | Conclusion |
|--------|------|---------------------------|-------------------|--------------|-------------------------|-------------|------------|
| V1.01 | 2026-07-18 | Unconditional Weekday | Global (Unconditional) | Welch t-test / WY | Westfall-Young (FWER) | N/A | Statistical Artifact. Project Pivoted. |
| V2.01 | [Pending] | Weekday × Volatility | VIX > 80th Pctile | Clustered Panel OLS | Westfall-Young | [Pending] | [Pending] |
| V2.02 | [Pending] | Weekday × Liquidity | Amihud > 90th Pctile | Clustered Panel OLS | Westfall-Young | [Pending] | [Pending] |
| V2.03 | [Pending] | Weekday × Sector Trend | Power/Nifty Beta > 1 | Clustered Panel OLS | Westfall-Young | [Pending] | [Pending] |
