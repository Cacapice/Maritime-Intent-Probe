"""
test_imports.py
===============
Cheapest possible regression guard: every analysis module imports without error
and the public probe symbols are present. Catches import-time breakage (a bad
edit, a missing dependency, a circular import) before any heavier test runs —
the "assert all top-level functions import cleanly" smoke from the review.

pytest-compatible and runnable standalone. Requires the repo's normal
environment (torch, sklearn, transformer_lens, wandb, …); intended for CI.
AGPL 3. Copyright © 2026 Katherine J. Ombrellaro.
"""
import importlib

import pytest

# Analysis / pipeline modules. (vault and monitor pull in wandb; stage_vault is a
# one-shot provisioning script — included so import-time errors still surface.)
MODULES = [
    "dtypes", "layers", "config", "stats", "sae", "probe",
    "environment", "collector", "patching", "experiment", "transfer",
    "surface_geometry", "bias_controls", "null_validator", "scale_probe",
    "notebook_setup", "vault",
]

PROBE_SYMBOLS = [
    "MassMeanProbe", "LogisticRegressionProbe", "ProbeResult",
    "_cv_auc", "_cv_auc_sae_per_fold", "_project_to_raw",
    "permutation_baseline_auc",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports_clean(name):
    importlib.import_module(name)


def test_probe_public_symbols_present():
    probe = importlib.import_module("probe")
    missing = [s for s in PROBE_SYMBOLS if not hasattr(probe, s)]
    assert not missing, f"probe.py missing expected symbols: {missing}"


if __name__ == "__main__":
    failures = []
    for m in MODULES:
        try:
            importlib.import_module(m)
            print(f"  PASS  import {m}")
        except Exception as e:                       # noqa: BLE001 - report all
            failures.append((m, repr(e)))
            print(f"  FAIL  import {m}: {e!r}")
    probe = importlib.import_module("probe")
    for s in PROBE_SYMBOLS:
        if not hasattr(probe, s):
            failures.append(("probe." + s, "missing"))
            print(f"  FAIL  probe.{s} missing")
    if failures:
        raise SystemExit(f"\n{len(failures)} import failure(s).")
    print("\nAll modules import cleanly.")
