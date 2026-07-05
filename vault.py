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
import wandb
from transformer_lens import HookedTransformer, HookedTransformerConfig

from config import VaultConfig

if TYPE_CHECKING:
    from sae import SparseAutoencoder

logger = logging.getLogger(__name__)


# ── Integrity ─────────────────────────────────────────────────────────────────

def sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


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
) -> HookedTransformer:
    """
    Full empirical load sequence:
      validate → verify integrity → load config → init model
      → inject weights → transfer device → track experiment
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
) -> Path:
    """
    Persist SAE weights and probe directions after each layer completes.
    Protects against Colab session disconnection mid-experiment.

    Saved files:
      <ckpt_dir>/saes/<layer>.pt        — SAE state dict per layer
      <ckpt_dir>/probe_directions.pt    — {layer: unit_direction} tensor dict
    """
    ckpt_dir = cfg.checkpoint_dir / version
    sae_dir  = ckpt_dir / "saes"
    sae_dir.mkdir(parents=True, exist_ok=True)

    for layer, sae in saes.items():
        safe_name = layer.replace(".", "_")
        torch.save(sae.state_dict(), sae_dir / f"{safe_name}.pt")

    torch.save(directions, ckpt_dir / "probe_directions.pt")
    logger.info("✓ Probe state saved: %d SAEs + directions → %s", len(saes), ckpt_dir)
    return ckpt_dir


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
    from sae import SparseAutoencoder

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
