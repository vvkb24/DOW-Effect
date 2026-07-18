# Implementation Principles

This document defines the strict engineering philosophy for this quantitative research repository.

1. **No Hidden Assumptions:** Every statistical function must explicitly log its assumptions. If normality is assumed, it must be documented in the docstring.
2. **No Hardcoded Constants:** Magic numbers (like `window=63`, `alpha=0.05`) must reside in `.yaml` config files, never hardcoded in `.py` files.
3. **No Silent Failures:** If an econometric model fails to converge, it must raise a loud exception, not silently return `NaN` or a poorly fit model.
4. **Literature Citing:** Every advanced statistical method (e.g., MBB, Petersen clustering, Westfall-Young) must include a comment citing the paper (Author, Year).
5. **Type Hints:** Every function must have complete Python type hints (`-> pd.DataFrame`, `-> float`).
6. **Tests:** Every module must have a corresponding test file in `tests/`.
7. **Reproducible Reports:** Every HTML or PDF report must be perfectly regenerable from the raw parquet data with a single command (`python scripts/run_phaseX.py`).
8. **Deterministic Seeds:** Every random process must use deterministic seeds (e.g., `np.random.seed(42)`).
9. **Metadata:** Every CSV or artifact output must include a timestamp and a Git commit hash indicating when it was generated.
10. **Structured Logs:** All modules must use the standard `logging` library, outputting to a central log file with timestamps and module names.
11. **Public Function Docs:** Every public function must document:
    - Inputs
    - Outputs
    - Assumptions
    - Failure modes
    - Complexity
    - References
