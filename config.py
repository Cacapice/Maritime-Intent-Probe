"""
config.py
AGPL 3. Copyright © 2026 Katherine J. Ombrellaro.
=========
VaultConfig: single source of truth for all file paths and run metadata.
Import this in every other module — nothing else should construct paths directly.

MODEL IDENTITY (2026-06-26)
---------------------------
`model_name` is now a first-class field. The weights/config/manifest vault lives
under a *per-model* subdirectory (`vault_dir = base_dir / model_slug`) so that,
e.g., Pythia-1.4B and Pythia-6.9B can be staged side-by-side without clobbering
each other. Selecting a model is a single assignment now —
`VaultConfig(model_name="EleutherAI/pythia-6.9b")` — and `load_model()` resolves
the right vault from it. Project-level artefacts (results, ledger, the
cross-model geometry JSONs) stay at `base_dir` so cross-model comparison cells
keep seeing both models in one place.

Migration note: the old layout put weights.pth flat in `base_dir`. The new
layout expects them under `base_dir/<model_slug>/`. Re-run stage_vault.py once
per model to populate the per-model vault (it now derives its paths from this
config, so they cannot drift).
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Optional


@dataclass
class VaultConfig:
    base_dir: Path = Path(
        os.getenv("VAULT_BASE_DIR", "/content/drive/MyDrive/secure_model_vault/maritime_intent_probe_v1")
    )
    # The model this vault holds. Overridable via env so stage_vault.py and the
    # notebook can agree on a single source of truth (VAULT_MODEL_NAME).
    model_name:        str = os.getenv("VAULT_MODEL_NAME", "EleutherAI/pythia-6.9b")
    weights_filename:  str = "weights.pth"
    config_filename:   str = "config.json"
    manifest_filename: str = "manifest.json"
    run_name:          str = "maritime_intent_probe_v1"
    checkpoint_version: Optional[str] = None
    seed: int = 42
    # FIX (2026-06-20): seed previously did not exist on this config at all --
    # _get_or_fit_sae/_make_sae_factory in experiment.py read cfg.seed via
    # getattr(..., None) and silently got None every time, making every SAE
    # fit (including the one underlying the locked BC1 gate) unseeded and
    # non-reproducible run-to-run. Default of 42 matches the value already
    # used elsewhere in this codebase for seeded checks (BC6's
    # compare_probes_multiseed uses seeds=[42, 123, 7]). See CHANGELOG.md.

    # ── Model identity ────────────────────────────────────────────────────────

    @property
    def model_slug(self) -> str:
        """Filesystem-safe model id, e.g. 'EleutherAI_pythia-6.9b'.
        Matches the MODEL_TAG.replace('/', '_') convention the analysis cells
        already use for the per-model geometry JSONs."""
        return self.model_name.replace("/", "_")

    @property
    def vault_dir(self) -> Path:
        """Per-model vault root holding weights/config/manifest/checkpoints."""
        return self.base_dir / self.model_slug

    # ── Vault artifact paths (per-model) ──────────────────────────────────────

    @property
    def weights_path(self) -> Path:
        if self.checkpoint_version:
            return self.vault_dir / "checkpoints" / self.checkpoint_version / self.weights_filename
        return self.vault_dir / self.weights_filename

    @property
    def config_path(self) -> Path:
        if self.checkpoint_version:
            return self.vault_dir / "checkpoints" / self.checkpoint_version / self.config_filename
        return self.vault_dir / self.config_filename

    @property
    def manifest_path(self) -> Path:
        return self.vault_dir / self.manifest_filename

    @property
    def checkpoint_dir(self) -> Path:
        return self.vault_dir / "checkpoints"

    # ── Project-level paths (shared across models) ────────────────────────────

    @property
    def ledger_path(self) -> Path:
        return self.base_dir / "run_ledger.jsonl"

    @property
    def artifacts(self) -> dict[str, Path]:
        return {"weights": self.weights_path, "config": self.config_path}

    # ── Results paths ─────────────────────────────────────────────────────────

    _RESULT_FILES: ClassVar[dict[str, str]] = {
        "null_control":       "null_control_result.json",
        "bias_diagnostics":   "bias_diagnostics.json",
        "transfer_results":   "transfer_results.json",
        "scale_probe":        "scale_probe_result.json",
        "causal_comparison":  "causal_comparison.json",
        "preregistration":    "PREREGISTRATION.md",
    }

    def result_path(self, name: str) -> Path:
        """Return the output path for a named result artefact (project-level)."""
        if name not in self._RESULT_FILES:
            raise KeyError(f"Unknown result name '{name}'. Valid: {list(self._RESULT_FILES)}")
        return self.base_dir / self._RESULT_FILES[name]

    # Convenience properties retained for backward compatibility
    @property
    def null_control_path(self)      -> Path: return self.result_path("null_control")
    @property
    def bias_diagnostics_path(self)  -> Path: return self.result_path("bias_diagnostics")
    @property
    def transfer_results_path(self)  -> Path: return self.result_path("transfer_results")
    @property
    def scale_probe_path(self)       -> Path: return self.result_path("scale_probe")
    @property
    def causal_comparison_path(self) -> Path: return self.result_path("causal_comparison")
    @property
    def preregistration_path(self)   -> Path: return self.result_path("preregistration")

    def validate(self, create: bool = False) -> None:
        if not self.vault_dir.exists():
            if create:
                self.vault_dir.mkdir(parents=True, exist_ok=True)
                self.checkpoint_dir.mkdir(exist_ok=True)
            else:
                raise FileNotFoundError(
                    f"Vault not found for model '{self.model_name}': {self.vault_dir}\n"
                    f"Run stage_vault.py (VAULT_MODEL_NAME='{self.model_name}') to provision it."
                )

        if missing := {k: v for k, v in self.artifacts.items() if not v.exists()}:
            raise FileNotFoundError(
                "Missing artifacts:\n" + "\n".join(f"  {k}: {v}" for k, v in missing.items())
            )

    def latest_checkpoint(self) -> Optional[str]:
        """Return the most recently modified checkpoint version name, or None."""
        if not self.checkpoint_dir.exists():
            return None
        versions = sorted(
            (d for d in self.checkpoint_dir.iterdir() if d.is_dir()),
            key=lambda d: d.stat().st_mtime,
        )
        return versions[-1].name if versions else None
