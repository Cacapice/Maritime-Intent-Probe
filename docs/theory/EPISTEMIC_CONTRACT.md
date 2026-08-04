# Epistemic Contract

This repository separates **evidence**, **interpretation**, and **permission to interpret**.

## Canonical BC1 statement

Semantic interpretation requires experimental identifiability. If an observationally equivalent model-blind statistic exists on the observed support, probe validity cannot establish construct validity.

## Status vocabulary

- `supported`, the stated interpretation is supported under declared assumptions.
- `blocked`, a named gate forbids the interpretation.
- `unidentified`, the experimental design does not identify the target contrast.
- `exploratory`, the result may guide research but is not confirmatory evidence.

## Publication rule

A result may show excellent predictive or causal behavior while remaining semantically unidentified. Reliability controls strengthen the claim that a pattern is real. They do not by themselves establish what the pattern means.

The canonical implementation is in `evidence_platform/epistemic.py`. Downstream reports should serialize `BC1Report.to_dict()` rather than reconstructing status from prose.


### Partial-witness interpretation

The repository-computed turn counter reaches AUC 1.000 on the preregistered multi-turn families and AUC 0.750 across all four families. The all-family value is therefore a partial model-blind witness, not a cleared null. Per-family diagnostics show that `fragmentation` and `priming` are perfectly separated by turn count while `semantic` and `encoding` are at chance. Those single-turn families are the first targets for the crossed 2×2 redesign.
