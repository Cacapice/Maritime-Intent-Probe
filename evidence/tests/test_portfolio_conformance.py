"""Conformance checks for the repository-local evidence runtime.

The runtime is bundled under ``evidence/runtime`` so a clean checkout installs
without relying on an unpublished package index. Domain research remains under
``science``; the shared contract remains isolated from adapters and experiments.
"""
from pathlib import Path

from evidence.adapter import Qualification, ScientificResult
from evidence.runtime import __version__

ROOT = Path(__file__).resolve().parents[2]
ADAPTER = ROOT / "evidence" / "adapter.py"
RUNTIME = ROOT / "evidence" / "runtime"


def test_contract_types_resolve_to_isolated_runtime():
    for cls in (ScientificResult, Qualification):
        assert cls.__module__.startswith("evidence.runtime"), cls.__module__


def test_runtime_is_grouped_outside_research_code():
    assert RUNTIME.is_dir()
    assert not (ROOT / "science" / "contract.py").exists()
    assert not (ROOT / "science" / "sovereign.py").exists()


def test_runtime_major_version_is_supported():
    assert int(__version__.split(".")[0]) == 1


def test_repository_module_holds_only_an_adapter():
    text = ADAPTER.read_text(encoding="utf-8")
    assert "class ScientificResult" not in text
    assert "class Qualification" not in text
    assert "EPISTEMIC_STRENGTH = " not in text
    assert "def from_" in text
    assert len(text.splitlines()) < 120
