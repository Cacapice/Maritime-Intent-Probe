"""Import smoke tests with explicit core and optional integration boundaries."""
import importlib
import importlib.util

import pytest

CORE_MODULES = [
    "science.dtypes", "science.layers", "science.config", "science.stats",
    "evidence_platform.epistemic", "science.sae", "science.probe",
    "science.environment", "science.surface_geometry",
    "evidence_platform.notebook_setup",
]

INTEGRATION_MODULES = {
    "science.collector": ("transformer_lens",),
    "science.patching": ("transformer_lens",),
    "science.experiment": ("transformer_lens", "wandb"),
    "science.transfer": ("transformer_lens",),
    "science.bias_controls": ("transformer_lens",),
    "science.null_validator": ("transformer_lens",),
    "science.scale_probe": ("transformer_lens",),
    "evidence_platform.vault": ("transformer_lens",),
}

PROBE_SYMBOLS = [
    "MassMeanProbe", "LogisticRegressionProbe", "ProbeResult",
    "_cv_auc", "_cv_auc_sae_per_fold", "_project_to_raw",
    "permutation_baseline_auc",
]


@pytest.mark.parametrize("name", CORE_MODULES)
def test_core_module_imports_clean(name):
    importlib.import_module(name)


@pytest.mark.parametrize("name,dependencies", INTEGRATION_MODULES.items())
def test_integration_module_imports_when_dependencies_available(name, dependencies):
    missing = [dep for dep in dependencies if importlib.util.find_spec(dep) is None]
    if missing:
        pytest.skip(f"optional integration dependencies unavailable: {', '.join(missing)}")
    importlib.import_module(name)


def test_probe_public_symbols_present():
    probe = importlib.import_module("science.probe")
    missing = [symbol for symbol in PROBE_SYMBOLS if not hasattr(probe, symbol)]
    assert not missing, f"science.probe missing expected symbols: {missing}"
