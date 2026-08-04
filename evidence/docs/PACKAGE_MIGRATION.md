# Migration to a single shared contract

## What changed

`qualification-contract` is now an installable package holding 1 implementation
of the publication contract. The 4 repositories depend on it and contribute only
adapters.

| | Before | After |
|---|---:|---:|
| `scientific_contract.py` per repository | ~370 lines | 48 to 49 lines |
| `sovereign_architecture.py` per repository | 256 lines | 17 lines |
| Implementations of `ScientificResult` | 4 | 1 |
| Duplicated lines across the portfolio | ~2,500 | 0 |

Each repository's `scientific_contract.py` re-exports the package, so every
existing import continues to work unchanged, and then defines its own adapter:
`from_restricted_modulus_result`, `from_evidence_profile`,
`from_fidelity_summary`, `from_bc1_report`.

## The guard was replaced, not kept

The interim drift guard hashed each vendored core and compared the hashes. It
was the right check while the core was duplicated and the wrong check once it is
not, so it has been inverted. Each repository now carries
`test_portfolio_conformance.py`, which asserts that:

- `ScientificResult` and `Qualification` resolve to `qualification_contract`,
  meaning they are re-exported rather than redefined;
- the local module contains no `class ScientificResult`, no `class
  Qualification` and no `EPISTEMIC_STRENGTH`, and is under 120 lines;
- the repository contributes at least 1 `from_` adapter;
- the repository does not declare its own `schema_version`, which belongs to the
  package.

Negative control: appending a local `class Qualification` to 1 repository fails
2 of those 4 tests. Removing it restores them.

`portfolio/core-contract.sha256` has been deleted. A pinned hash of a file that
no longer exists in 4 copies would be a stale artifact asserting a property
nobody is checking.

## Verification

| | |
|---|---|
| Inferential Fidelity Framework | 174 passed |
| Benchmark Stewardship | 145 passed |
| Bayesian Inferential Fidelity | 57 passed |
| Maritime Intent Probe | 76 passed, 8 skipped |
| Package importable, schema | 1.1 |

## One residual

Maritime Intent Probe has no `pyproject.toml`, so its dependency on
`qualification-contract` is not declared anywhere a package manager can read. It
imports the package successfully because the package is installed in the
environment, which is not the same as depending on it. Adding a `pyproject.toml`
to that repository is the outstanding item.
