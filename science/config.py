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
    # BC1's main null-control result. A FIELD, like the filenames above, so the
    # name can be corrected without editing the property. Default matches the
    # file null_validator.py has always written to base_dir alongside its four
    # other flat-named outputs (null_control_by_encoding_variant[_normalized],
    # null_control_normalized_comparison, positive_control_intent_vs_random) --
    # chosen to adopt the existing artifact rather than orphan it under a new
    # name. VERIFY against base_dir if BC1 results predate this field.
    null_control_filename: str = "null_control_result.json"
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

    def result_path(self, filename: str) -> Path:
        """Project-level result, MODEL-TAGGED.

        base_dir keeps results where cross-model comparison cells can see every
        model at once; the slug in the FILENAME is what stops a second model's run
        overwriting the first's. This is the same convention the geometry cells
        already use via MODEL_TAG.replace("/", "_").

        Why tagging is not optional here: null_validator.py wrote five flat,
        untagged filenames, so running BC1 on 6.9B silently overwrote the 1.4B
        result IN PLACE -- and Amendment 2 exists precisely because those two
        models disagree on this gate. The clobbering was invisible: same path,
        same schema, no error, just different numbers under the old name.

            result_path("null_control_result.json")
            -> base_dir/null_control_result_EleutherAI_pythia-6.9b.json

        Pre-tagging artifacts keep their old untagged names and are NOT deleted;
        they are simply no longer written to. Nothing reads them (every site in
        null_validator.py is write-only), so nothing downstream breaks -- but note
        an old file cannot be attributed to a model from its name alone, which is
        the problem this method exists to end.
        """
        p = Path(filename)
        return self.base_dir / f"{p.stem}_{self.model_slug}{p.suffix}"

    @property
    def null_control_path(self) -> Path:
        """BC1 null-control result (null_validator.validate_null_control).

        RESTORED 2026-07-17. null_validator.py has always written here, but the
        property was lost when this config moved to the per-model layout, so BC1
        raised AttributeError at line 688 -- at WRITE time, after the whole
        validation had already run and the compute was spent.

        base_dir, not vault_dir: it is a RESULT, and config.py's convention keeps
        results project-level so cross-model cells see every model in one place.

        MODEL-TAGGED via result_path(), as are null_validator.py's other four
        outputs -- all five together, so none is the odd one out. Before this,
        every one was a flat untagged name and a second model's run overwrote the
        first's result in place; Amendment 2 exists because 1.4B and 6.9B disagree
        on exactly this gate.
        """
        return self.result_path(self.null_control_filename)

    @property
    def artifacts(self) -> dict[str, Path]:
        """Required model artifacts keyed by logical name."""
        return {
            "weights": self.weights_path,
            "config": self.config_path,
        }

    def validate(self, create: bool = False) -> None:
        """Validate that the configured model vault is complete.

        When ``create=True``, this method creates only the expected directory
        structure. It never downloads, converts, or generates model artifacts.
        """

        if not self.vault_dir.exists():
            if create:
                self.vault_dir.mkdir(parents=True, exist_ok=True)
            else:
                raise FileNotFoundError(
                    f"Vault not found for model '{self.model_name}': "
                    f"{self.vault_dir}\n"
                    "Provision it by running stage_vault.py with:\n"
                    f"  VAULT_MODEL_NAME='{self.model_name}'"
                )

        if create:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        missing = {
            name: path
            for name, path in self.artifacts.items()
            if not path.exists()
        }

        if not missing:
            return

        existing = {
            name: path
            for name, path in self.artifacts.items()
            if path.exists()
        }

        message = [
            f"Vault is incomplete for model '{self.model_name}'.",
            f"Vault directory: {self.vault_dir}",
            "",
            "Missing artifacts:",
            *[
                f"  {name}: {path}"
                for name, path in missing.items()
            ],
        ]

        if existing:
            message.extend(
                [
                    "",
                    "Existing artifacts:",
                    *[
                        f"  {name}: {path}"
                        for name, path in existing.items()
                    ],
                ]
            )

        message.extend(
            [
                "",
                "Provision or repair the vault by running stage_vault.py with:",
                f"  VAULT_MODEL_NAME='{self.model_name}'",
                "",
                "validate(create=True) creates directories only; "
                "it does not create model artifacts.",
            ]
        )

        raise FileNotFoundError("\n".join(message))
