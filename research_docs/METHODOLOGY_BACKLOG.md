# Methodology Backlog

This document stores future research ideas, expansions, alternative methods, and feature definitions that are slated for exploration under the Conditional Alpha Discovery framework.

## 1. New Interaction Terms
- `Weekday × Institutional Flow (FII/DII Net)`
- `Weekday × Gamma Exposure (GEX) Regime`
- `Weekday × Intermarket Spreads (e.g., Nifty Power vs Nifty Metal)`

## 2. New Regime Definitions
- **Interest Rate Regime:** Rising vs. Falling Repo Rate cycles (RBI).
- **Inflation Regime:** CPI above/below target bands.
- **Liquidity Regime:** Banking system liquidity deficit vs. surplus.

## 3. New Feature Families
- **Microstructure:** Order imbalance proxies, tick-level bid-ask spread proxies.
- **Alternative Data:** Weather anomalies (impact on power demand), coal stockpile shortages.

## 4. Alternative Model Classes
- **Hidden Markov Models (HMM):** For unsupervised regime detection prior to interaction testing.
- **Causal Inference Models:** DoWhy / Double Machine Learning to isolate the causal impact of event proximity on weekday returns.

## 5. Alternative Structural-Break Methods
- **Pruned Exact Linear Time (PELT):** For faster and more robust multivariate change-point detection across interaction vectors.
