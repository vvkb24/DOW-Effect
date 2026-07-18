# Code Review Standard

Defines the minimum quality required before code is merged into the `main` branch. Every Pull Request must satisfy this checklist.

- [ ] **Correctness:** Does the code do exactly what the Research Contract specifies?
- [ ] **Scientific Correctness:** Is the data processing free of survivorship, lookahead, and leakage biases?
- [ ] **Statistical Correctness:** Are the degrees of freedom correct? Are tails properly handled?
- [ ] **Econometric Correctness:** Are matrix inversions stable? Are standard errors clustered correctly?
- [ ] **Financial Correctness:** Do the calculated returns exactly match theoretical compounding formulas?
- [ ] **Performance:** Are loops vectorized using `pandas`/`numpy`? (No `iterrows` allowed).
- [ ] **Maintainability:** Are functions under 50 lines? Is cyclomatic complexity low?
- [ ] **Documentation:** Do all public functions have docstrings matching the Implementation Principles?
- [ ] **Testing:** Does the PR add unit tests for the new logic? Is coverage > 90% for the new module?
- [ ] **Reproducibility:** Does running the script end-to-end yield the exact same output hashes?
- [ ] **Configuration:** Are all new variables pushed to `config/*.yaml`?
