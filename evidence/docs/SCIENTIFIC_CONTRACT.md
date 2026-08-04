# Scientific Output Contract

These repositories share a common machine-readable publication contract under the program **Qualification Before Interpretation**.

Every publication result declares:

1. **Estimand** — what quantity or epistemic claim is being represented.
2. **Estimate kind** — exact value, point estimate, Monte Carlo mean, empirical quantile, certified floor, lower bound, evidence profile, or identifiability gate.
3. **Uncertainty** — interval, confidence level, sample count, standard error, and method when estimable.
4. **Assumptions and qualifications** — conditions required for interpretation, including censoring, saturation, finite-draw limitations, and information-basis restrictions.
5. **Publication status** — publishable, qualified, exploratory, or blocked.
6. **Supported and unsupported conclusions** — explicit inference boundaries.
7. **Evidence graph** — a machine-readable path from observation through statistic and qualification to inference.
8. **Computational provenance** — generation time, commit, Python version, platform, and seed.

The dependency-light `scientific_contract` module provides `ScientificResult`, `Uncertainty`, `EvidenceGraph`, and `publish_bundle`. Repository-specific adapters translate native result objects without replacing their richer domain APIs.

A deterministic bundle contains:

```text
scientific_result.json
evidence_graph.json
provenance.json
```

This schema does not make unlike estimands interchangeable. It standardizes how their evidential status is declared.
