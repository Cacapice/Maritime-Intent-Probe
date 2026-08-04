# Rectification of the limitations raised

## Retracted

**The privacy model is enforced.** I recorded it as declarative only. That was
wrong, and testing it showed why: `SovereignEvidenceEnvelope.evaluate` blocks on
a purpose outside the owner-declared boundary, on direct identifiers, on high or
unknown disclosure risk, on a non-publishable underlying result, and on missing
`data_owner` or `publication_authority` approval. It qualifies rather than blocks
on non-minimized or non-aggregate payloads and on moderate residual risk. The
constructor refuses any envelope whose payload carries `raw_records`,
`raw_data`, `direct_identifiers` or `row_level_data`. `strict_release` raises on
a blocked envelope.

I had read the documents and not exercised the code. The finding is withdrawn.

## Fixed

### 1. Contract drift across 4 copies

`scientific_contract.py` is 360 lines of shared core duplicated across 4
repositories, and the copies had already diverged before my earlier patch.

An adapter marker now separates the shared core from repository-specific
adapters, so the 2 parts can be checked differently: the core must be identical,
and the adapters must differ. A repository whose adapter section is empty has not
integrated the contract, and that is now an error rather than an oversight.

The canonical core hash is pinned in `portfolio/core-contract.sha256`, and each
repository carries `test_portfolio_conformance.py`, which fails if its core
drifts. Negative control: inserting a single comment into one repository's core
fails that repository's suite.

### 2. The portfolio verifier verified nothing

The previous version asserted 2 manifest fields and printed `"repositories": 4`
as a literal. It opened no repository, so it would have passed unchanged if all 4
had drifted to different schemas.

It now reads every repository and checks 7 things: that each declared repository
exists, that the shared core is byte-identical across all of them, that the core
matches the pinned canonical hash, that the adapter sections differ, that each
repository's declared `schema_version` matches the manifest's claim, that
`sovereign_architecture.py` has not diverged, and that a `phase_1` claim in the
manifest is actually handled by the repository it names.

Negative controls: drifting one core is caught; changing one repository's
declared schema version is caught and reported separately from the hash
divergence.

### 3. The default research phase granted the strongest claim

`research_phase` defaulted to `mature_method` in 8 places. A default that awards
the strongest phase to any caller who does not think about it runs against the
doctrine the schema exists to serve.

The default is now `exploratory`, with the reasoning recorded at the declaration.
A caller who wants a stronger phase must say so.

## Verification

| | |
|---|---|
| Inferential Fidelity Framework | 172 passed |
| Benchmark Stewardship | 71 passed |
| Bayesian Inferential Fidelity | 55 passed |
| Maritime Intent Probe | 26 passed |
| Portfolio conformance | valid, 0 errors |

## Still open

The core is duplicated, and it is now *checked* rather than *shared*. A drift is
caught, but 4 copies must still be edited in step, and the conformance test will
correctly fail the other 3 until they are. The proper remedy remains a single
installable `qualification-contract` package with the repositories depending on
it and holding only their adapters. The pinned hash is a guard, not a fix.
