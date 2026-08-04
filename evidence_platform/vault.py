"""
vault.py
AGPL 3. Copyright © 2026 Katherine J. Ombrellaro.
========
Infrastructure for model integrity, experiment tracking, resilient loading,
and checkpoint management. Stable once working — should not need changes
during normal experimental iteration.
"""

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import torch
try:
    import wandb
except Exception:  # optional: init_tracking already degrades gracefully
    wandb = None  # wandb.init access below then raises and is caught,
                  # logging the 'W&B unavailable' warning as designed.
from transformer_lens import HookedTransformer, HookedTransformerConfig

from science.config import VaultConfig

if TYPE_CHECKING:
    from science.sae import SparseAutoencoder

logger = logging.getLogger(__name__)


# ── Integrity ─────────────────────────────────────────────────────────────────

def sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def vault_processing(cfg: VaultConfig) -> Optional[dict]:
    """Read the `_processing` record stage_vault.py wrote into the manifest.

    Returns None for a LEGACY vault staged before the mode was recorded, whose
    mode is therefore unknown from the manifest alone -- see
    _check_processing_mode for how far config.json can be used to infer it.
    """
    if not cfg.manifest_path.exists():
        return None
    try:
        return json.loads(cfg.manifest_path.read_text()).get("_processing")
    except json.JSONDecodeError:
        return None


def _check_processing_mode(cfg: VaultConfig, config_dict: dict, expect: Optional[str]) -> None:
    """Refuse a vault staged under a different weight-processing mode.

    WHY THIS EXISTS. TransformerLens's from_pretrained() rewrites weights by
    default (fold_ln, center_writing_weights, center_unembed, fold_value_biases);
    from_pretrained_no_processing() does not. fold_ln folds each LayerNorm's
    learned scale into the following linear layer, so blocks.N.hook_resid_post
    holds a DIFFERENT vector under the two modes. A vault staged one way and
    loaded by a notebook that builds the model the other way yields two different
    models, no error, and silently incomparable probe directions, deltas and AUCs.
    This project ran in exactly that state: the vault staged with from_pretrained
    while the notebook loaded with from_pretrained_no_processing.

    The manifest's `_processing.mode` is authoritative. config.json's
    normalization_type is a cross-check that catches fold_ln only ("LNPre" =
    folded, "LN" = raw); the other three flags leave no trace in the config, which
    is exactly why the manifest record is needed.
    """
    recorded = vault_processing(cfg)
    norm = config_dict.get("normalization_type")
    inferred_fold_ln = None if norm is None else (norm == "LNPre")

    if recorded is None:
        msg = (
            "Vault manifest has no `_processing` record (staged before the mode "
            "was tracked), so its weight-processing mode cannot be confirmed. "
            f"config.json normalization_type={norm!r} implies fold_ln="
            f"{inferred_fold_ln} -- but center_writing_weights, center_unembed "
            "and fold_value_biases leave no trace in the config and remain "
            "unknown. Re-stage with stage_vault.py to record the mode."
        )
        if expect is not None:
            raise RuntimeError(
                msg + f"\n\nCalled with expect_processing={expect!r}, which cannot "
                "be verified against an unrecorded vault. Re-stage, or drop "
                "expect_processing to load without the check (accepting that the "
                "model may not match what the notebook builds)."
            )
        logger.warning(msg)
        return

    mode = recorded.get("mode")
    if inferred_fold_ln is not None and recorded.get("fold_ln") is not None:
        if bool(recorded["fold_ln"]) != inferred_fold_ln:
            raise RuntimeError(
                f"Vault is internally inconsistent: manifest says fold_ln="
                f"{recorded['fold_ln']} but config.json normalization_type={norm!r} "
                f"implies fold_ln={inferred_fold_ln}. The manifest and the config "
                "describe different models; re-stage this vault."
            )
    if expect is not None and mode != expect:
        raise RuntimeError(
            f"Weight-processing mismatch: this vault was staged as {mode!r} "
            f"({recorded.get('staged_call')}), but the caller expects {expect!r}.\n"
            "These are DIFFERENT MODELS -- fold_ln rewrites LayerNorm into the "
            "following linear layer, so hook_resid_post differs, and every probe "
            "direction, delta and AUC computed from them is incomparable.\n"
            f"Either re-stage with VAULT_PROCESSING={expect}, or load with "
            f"expect_processing={mode!r} if that is genuinely the model you want."
        )
    logger.info("\u2713 Weight processing: %s (%s)", mode, recorded.get("staged_call"))


def verify_manifest(cfg: VaultConfig) -> dict[str, str]:
    """
    Compare current artifact hashes against stored manifest.
    Writes a new manifest if none exists (first run).
    Returns the hash dict for downstream logging.
    """
    hashes = {name: sha256(path) for name, path in cfg.artifacts.items()}

    if not cfg.manifest_path.exists():
        cfg.manifest_path.write_text(json.dumps(hashes, indent=2))
        logger.warning("No manifest found — wrote initial manifest. Verify before production use.")
        return hashes

    stored    = json.loads(cfg.manifest_path.read_text())
    corrupted = {k: v for k, v in hashes.items() if stored.get(k) != v}

    if corrupted:
        raise RuntimeError(
            "Integrity check failed — hash mismatch:\n"
            + "\n".join(f"  {k}: expected {stored[k][:12]}… got {v[:12]}…"
                        for k, v in corrupted.items())
        )

    logger.info("✓ Manifest verified: all hashes match.")
    return hashes


# ── Experiment Tracking ───────────────────────────────────────────────────────

def init_tracking(cfg: VaultConfig, hashes: dict[str, str], params: int) -> None:
    """Log run metadata to W&B and append to the local ledger."""
    run_meta = {
        "run_name":           cfg.run_name,
        "model_name":         cfg.model_name,
        "checkpoint_version": cfg.checkpoint_version or "base",
        "params":             params,
        "hashes":             hashes,
        "timestamp":          datetime.now(timezone.utc).isoformat(),
    }

    try:
        wandb.init(project="maritime_intent_probe_audit", name=cfg.run_name, config=run_meta)
        wandb.log({"param_count": params, **{f"hash_{k}": v[:12] for k, v in hashes.items()}})
        logger.info("✓ W&B run initialized.")
    except Exception as e:
        logger.warning("W&B unavailable — skipping remote tracking: %s", e)

    with open(cfg.ledger_path, "a") as f:
        f.write(json.dumps(run_meta) + "\n")
    logger.info("✓ Run logged to ledger: %s", cfg.ledger_path.name)


# ── Resilient Weight Loading ──────────────────────────────────────────────────

def load_state_dict_safe(
    model: HookedTransformer,
    weights_path: Path,
    retries: int = 2,
) -> None:
    """Load state dict with retry + exponential backoff for transient I/O failures."""
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict, strict=True)
            return
        except (RuntimeError, EOFError, OSError) as e:
            last_exc = e
            if attempt < retries:
                logger.warning("Load attempt %d/%d failed — retrying: %s", attempt, retries, e)
                time.sleep(2 ** attempt)

    if isinstance(last_exc, RuntimeError):
        raise RuntimeError(f"Weight/config mismatch in '{weights_path.name}': {last_exc}") from last_exc
    raise OSError(f"Corrupt or unreadable weights after {retries} attempts: {last_exc}") from last_exc


def load_model(
    cfg: VaultConfig,
    device: str = "cuda",
    verify_integrity: bool = True,
    track: bool = True,
    expect_processing: Optional[str] = None,
) -> HookedTransformer:
    """
    Full empirical load sequence:
      validate → verify integrity → check processing mode → load config
      → init model → inject weights → transfer device → track experiment

    expect_processing: "none" | "standard" | None. The weight-processing mode the
        CALLER requires (see stage_vault.py's VAULT_PROCESSING). Passing it turns
        a mode mismatch into a loud error; leaving it None only logs the mode.
        Pass it whenever the same session also builds a model directly via
        from_pretrained*(), since that is where the two silently diverge:
            notebook uses from_pretrained_no_processing -> expect_processing="none"
            notebook uses from_pretrained                -> expect_processing="standard"
    """
    cfg.validate()
    hashes = verify_manifest(cfg) if verify_integrity else {}

    try:
        config_dict = json.loads(cfg.config_path.read_text())
        # Coerce dtype string → torch.dtype; HookedTransformerConfig.from_dict no longer
        # performs this conversion, so torch.empty(..., dtype=str) would TypeError at init.
        # NOTE: model.cfg.to_dict() serialises the dtype as e.g. "torch.bfloat16"
        # (str(torch.bfloat16) == "torch.bfloat16"), so we strip the "torch." prefix
        # and normalise aliases. Without this, a bf16/fp16 vault (how 6.9B is staged,
        # to fit memory) would miss the map, fall back to float32, init the model in
        # fp32, and then upcast the loaded half-precision weights — doubling memory and
        # OOMing on large models. Keep this map and the stage dtype in lock-step.
        raw_dtype = config_dict.get("dtype")
        if isinstance(raw_dtype, str):
            _DTYPE_MAP = {
                "float32":  torch.float32, "float": torch.float32, "fp32": torch.float32,
                "float16":  torch.float16, "half":  torch.float16, "fp16": torch.float16,
                "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
            }
            key = raw_dtype.replace("torch.", "").strip().lower()
            if key not in _DTYPE_MAP:
                logger.warning(
                    "Unrecognised dtype '%s' in config — defaulting to float32. "
                    "If this is a half-precision vault, the load will upcast and may OOM.",
                    raw_dtype,
                )
            config_dict["dtype"] = _DTYPE_MAP.get(key, torch.float32)
        model_cfg   = HookedTransformerConfig.from_dict(config_dict)
    except (json.JSONDecodeError, KeyError) as e:
        raise ValueError(f"Malformed config ({cfg.config_path.name}): {e}") from e

    # After the config parses (its normalization_type is the cross-check) and
    # before any weights are read: a mode mismatch should cost nothing.
    _check_processing_mode(cfg, config_dict, expect_processing)

    model = HookedTransformer(model_cfg)   # device= removed: unsupported in current TL versions
    load_state_dict_safe(model, cfg.weights_path)

    torch.cuda.empty_cache()
    model.to(device)

    params = sum(p.numel() for p in model.parameters())
    if track:
        init_tracking(cfg, hashes, params)

    _dtype = next(model.parameters()).dtype
    logger.info("✓ %s (%s) on %s | dtype: %s | params: %s",
                cfg.run_name, cfg.model_name, device, _dtype, f"{params:,}")
    return model


# ── Checkpoint Utilities ──────────────────────────────────────────────────────

def save_checkpoint(model: HookedTransformer, cfg: VaultConfig, version: str) -> Path:
    """
    Save a versioned model checkpoint and update the manifest.
    version: e.g. "epoch_04", "step_10000", "ablation_no_mlp"
    """
    ckpt_dir = cfg.checkpoint_dir / version
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    weights_out = ckpt_dir / cfg.weights_filename
    torch.save(model.state_dict(), weights_out)

    config_out = ckpt_dir / cfg.config_filename
    config_out.write_text(cfg.config_path.read_text())

    hashes = {"weights": sha256(weights_out), "config": sha256(config_out)}
    (ckpt_dir / cfg.manifest_filename).write_text(json.dumps(hashes, indent=2))

    logger.info("✓ Checkpoint saved: %s", version)
    return ckpt_dir


def save_probe_state(
    saes: dict[str, "SparseAutoencoder"],
    directions: dict[str, torch.Tensor],
    cfg: VaultConfig,
    version: str,
    gate_status: dict | None = None,
) -> Path:
    """
    Persist SAE weights and probe directions after each layer completes.
    Protects against Colab session disconnection mid-experiment.

    Saved files:
      <ckpt_dir>/saes/<layer>.pt        — SAE state dict per layer
      <ckpt_dir>/probe_directions.pt    — {layer: unit_direction} tensor dict
      <ckpt_dir>/gate_status.json       — BC1 verdict, when gate_status is given

    PERSISTENCE IS NOT LICENSING.
    ------------------------------
    Saving state and being ENTITLED TO USE IT are different things, and until now
    they were conflated by execution order alone: BC1 raising aborted Cell 3
    before the save, so a failed gate left no checkpoint, so the ~15 cells calling
    restore_analysis_state could not run. That protection was ACCIDENTAL — a
    side effect of where the raise sat — and it had a real cost: an hour of
    collection and per-layer SAE fits was destroyed on every failed run, which is
    exactly the state you most need to keep in order to DIAGNOSE the failure.

    Passing `gate_status` (the dict validate_null_control returns) records the
    verdict ALONGSIDE the checkpoint. restore_analysis_state then refuses to load
    state whose gate failed unless the caller explicitly opts in. So the
    checkpoint can be written before, after, or independently of the gate, and the
    gate still governs what the state may be USED FOR — which is what it was
    always meant to control. Ordering stops being load-bearing.

    Args:
        gate_status: BC1's result dict (keys: passed, worst_auc, deviation,
            tolerance, scope, message, ...). None writes no status file, and
            restore_analysis_state will then warn that the state is unattributed
            rather than silently trusting it.
    """
    ckpt_dir = cfg.checkpoint_dir / version
    sae_dir  = ckpt_dir / "saes"
    sae_dir.mkdir(parents=True, exist_ok=True)

    for layer, sae in saes.items():
        safe_name = layer.replace(".", "_")
        torch.save(sae.state_dict(), sae_dir / f"{safe_name}.pt")

    torch.save(directions, ckpt_dir / "probe_directions.pt")

    if gate_status is not None:
        passed = bool(gate_status.get("passed"))
        (ckpt_dir / "gate_status.json").write_text(json.dumps({
            "passed":     passed,
            "worst_auc":  gate_status.get("worst_auc"),
            "deviation":  gate_status.get("deviation"),
            "tolerance":  gate_status.get("tolerance"),
            "scope":      gate_status.get("scope"),
            "message":    gate_status.get("message"),
            "by_source_class": gate_status.get("by_source_class"),
            "model_name": cfg.model_name,
            "saved_at":   datetime.now(timezone.utc).isoformat(),
        }, indent=2, default=str))
        if not passed:
            logger.warning(
                "Probe state saved with a FAILED gate (%s). It is kept for "
                "DIAGNOSIS; restore_analysis_state will refuse it unless called "
                "with allow_failed_gate=True.", gate_status.get("scope"),
            )
    logger.info("✓ Probe state saved: %d SAEs + directions → %s", len(saes), ckpt_dir)
    return ckpt_dir


def probe_state_gate(cfg: VaultConfig, version: str) -> dict | None:
    """Read the gate verdict stored beside a probe checkpoint, or None when the
    checkpoint predates gate recording (its provenance is then unknown)."""
    p = cfg.checkpoint_dir / version / "gate_status.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def load_probe_state(
    cfg: VaultConfig,
    version: str,
    d_model: int,
    n_features: int,
    l1_coeff: float = 2e-4,
) -> tuple[dict[str, "SparseAutoencoder"], dict[str, torch.Tensor]]:
    """
    Restore SAE weights and probe directions from a saved checkpoint.
    Use after a Colab session drop to resume without refitting.
    """
    from science.sae import SparseAutoencoder

    ckpt_dir   = cfg.checkpoint_dir / version
    sae_dir    = ckpt_dir / "saes"
    directions = torch.load(ckpt_dir / "probe_directions.pt", weights_only=True)
    saes: dict[str, SparseAutoencoder] = {}

    for path in sae_dir.glob("*.pt"):
        layer = path.stem.replace("_", ".", 2)
        sae   = SparseAutoencoder(d_model, n_features, l1_coeff)
        sae.load_state_dict(torch.load(path, weights_only=True))
        saes[layer] = sae

    logger.info("✓ Probe state restored: %d SAEs + directions from %s", len(saes), ckpt_dir)
    return saes, directions
