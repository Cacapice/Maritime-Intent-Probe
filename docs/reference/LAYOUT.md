# Repository layout

- `README.md` — scientific entry point and Phase 1 claim boundary.
- `science/` — research implementation: stimuli, probes, controls, geometry, patching, statistics, and the dependency-light model-blind witness.
- `evidence_platform/` — repository-local qualification, vaulting, and reproducibility support.
- `evidence/` — adapter and sovereign-publication integration with the shared `high-trust-evidence` package.
- `docs/` — theory, research narrative, figure governance, and reference material.
- `figures/` — registered publication figures and sidecars.
- root `test_*.py` — scientific and repository-level regression contracts.

Research code is grouped under `science/`; publication infrastructure is deliberately separated so it does not crowd the methodological contribution.
