
## Trust hardening v1.3

- Added executable repository-owned adapter qualification policies and negative controls for under-qualification.
- Made signed manifests mandatory by default for owner-held, derived-owner, internal, controlled, and aggregate releases.
- Bound trust decisions to exact policy bytes and added explicit policy-migration records with mandatory re-evaluation.
- Kept figure inventories enforced in CI; Maritime remains locked to Phase 1 until the crossed design is complete.


## Integrated trust architecture

- Merged figure governance, qualification algebra, and sovereign emission controls into one release path.
- Publication status is derived from a monotone qualification transition system.
- Trust decisions are bound to exact result/envelope digests, purpose, classification, and policy version.
- Added optional authenticated manifests for enterprise publication mode.

## Sovereign Evidence Architecture

- Added an enterprise-level sovereign-data and high-trust privacy release envelope.
- Added purpose limitation, aggregate-only disclosure, owner/custodian provenance, dual publication approval, and integrity-verifiable sovereign bundles.
- Kept scientific qualification distinct from privacy approval and final human decision authority.

## Publication contract v1.1

- Added versioned JSON Schemas, qualification monotonicity, SHA-256 bundle manifests, expanded provenance, deterministic verification CLI, and research-phase declaration.
- Maritime is explicitly Phase 1 only.

# Changelog

- Added `FIGURE_STANDARD.md` and `FIGURE_MAP.md`: figures now have a declared question, observation, interpretation, inference status, verified scales, and direct-label preference.

## Unreleased

- Interprets the all-family turn-count AUC of 0.750 as a partial witness and adds per-family diagnostics that localize the residual gap to `semantic` and `encoding`, making the redesign priority explicit.

- Elevated BC1 to the canonical repository-level construct-validity criterion.
- Added dependency-free machine-readable epistemic contracts and strict interpretation gating.
- Added evidence/interpretation/limit presentation and a repository scientific contract.
- Added regression protection against unresolved Git merge markers.
- Resolved merge artifacts present in the supplied source archive.
- Consolidated positioning and fellowship documents around the canonical BC1 statement.

## Witness-claim integration
- Added `compute_model_blind_witness.py`, which generates payloads and computes the turn-count witness in-repository.
- Scoped the AUC 1.000 claim to the preregistered multi-turn attack families and reports the all-family result alongside it.
- Added full dependency-light tests for `science/witness_test.py`.
- Corrected the README minimal-pair quotation to match `science/environment.py`.
- Added default-branch CI for compilation, methodological tests, and witness-artifact generation.

- Refactored the model-blind witness path to depend only on `science/payload_templates.py`
 and `science/witness_test.py`. Importing or running it no longer imports PyTorch or the
 model-facing environment. Added a fresh-interpreter regression guard.

- Expanded dependency-light CI, added computed witness-contract assertions, citation metadata, and a standard license filename.

- Added the cross-repository ScientificResult schema, evidence graph, provenance capture, and deterministic publication bundles.
