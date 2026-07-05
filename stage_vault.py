"""
stage_vault.py — one-time creation of the secure model vault on Drive.
AGPL 3. Copyright © 2026 Katherine J. Ombrellaro.
=======================================================================
Run ONCE in Colab (after mounting Drive, and after Cell 1/1a installs the repo)
before Cell 2 of the notebook. This is NOT part of the experiment run; it
provisions the integrity-verified vault that vault.load_model() reads.
load_model() never downloads — by design, so its hash manifest can guarantee the
weights are untampered. This script is the trusted step that puts known weights
into the vault and records their hashes.

MODEL SELECTION (2026-06-26)
----------------------------
Default model is now Pythia-6.9B and the vault is PER-MODEL: paths come straight
from VaultConfig (`cfg.vault_dir = base_dir / model_slug`), so staging 6.9B does
NOT clobber an existing 1.4B vault — each lives in its own subdirectory. To stage
a different size, set VAULT_MODEL_NAME:

    VAULT_MODEL_NAME='EleutherAI/pythia-1.4b' python stage_vault.py
    VAULT_MODEL_NAME='EleutherAI/pythia-6.9b' python stage_vault.py   # default

PRECISION / MEMORY
------------------
6.9B is staged in bfloat16 (~13.8 GB on disk) rather than float32 (~27.6 GB).
That halves the Drive footprint, the CPU RAM needed here, and the VRAM needed in
Cell 2. vault.load_model() reads the dtype from config.json and inits the model
in that dtype, so the stage dtype IS the inference dtype — keep them consistent.
Override with VAULT_DTYPE ('bfloat16' | 'float16' | 'float32').

Memory expectations for the default (6.9B, bf16):
  • Staging here (CPU): needs a HIGH-RAM runtime (~25-30 GB peak). The standard
    ~12 GB Colab runtime will OOM — use an A100/high-RAM runtime.
  • Loading in Cell 2 (GPU): ~14 GB weights + activations. Use an A100-class
    (>=40 GB) runtime; a T4/L4 will OOM.

What it does:
  1. Loads the selected model via TransformerLens in the chosen dtype
     (HF download happens HERE, once).
  2. Writes weights.pth as a TransformerLens-format state_dict (correct key names
     for HookedTransformer.load_state_dict(strict=True) — NOT raw HF keys).
  3. Writes config.json from the model's HookedTransformerConfig (the exact dict
     HookedTransformerConfig.from_dict() expects in load_model(), including the
     dtype string that load_model() coerces back to a torch.dtype).
  4. Writes manifest.json = streaming SHA-256 of weights.pth and config.json,
     computed AFTER writing, matching vault.sha256() byte-for-byte so
     verify_manifest() passes on the first real load.

After running this, Cell 2's load_model(CFG, ...) will validate → verify → load
cleanly. If you ever re-stage the SAME model, delete that model's subdirectory
first so a stale manifest isn't compared against new files.
"""
import os, json, hashlib

import torch
from transformer_lens import HookedTransformer

# Single source of truth for paths — never reconstruct them by hand here, or the
# manifest/layout can drift from what load_model() expects.
from config import VaultConfig

MODEL_NAME = os.getenv("VAULT_MODEL_NAME", "EleutherAI/pythia-6.9b")
DTYPE_NAME = os.getenv("VAULT_DTYPE", "bfloat16")

_DTYPE_MAP = {
    "float32":  torch.float32, "fp32": torch.float32,
    "float16":  torch.float16, "fp16": torch.float16,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
}


def sha256(path, chunk_size: int = 1 << 20) -> str:
    """Identical to vault.sha256 — streaming SHA-256, 1 MiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if DTYPE_NAME.replace("torch.", "").lower() not in _DTYPE_MAP:
        raise SystemExit(f"Unknown VAULT_DTYPE='{DTYPE_NAME}'. Valid: {sorted(_DTYPE_MAP)}")
    torch_dtype = _DTYPE_MAP[DTYPE_NAME.replace("torch.", "").lower()]

    # Paths come from the same config the notebook/load_model use, keyed to this
    # model. checkpoint_version is None here, so weights/config/manifest land flat
    # in the per-model vault dir (base_dir / model_slug).
    cfg = VaultConfig(model_name=MODEL_NAME)
    cfg.vault_dir.mkdir(parents=True, exist_ok=True)

    weights_path  = cfg.weights_path
    config_path   = cfg.config_path
    manifest_path = cfg.manifest_path

    if weights_path.exists() or config_path.exists() or manifest_path.exists():
        raise SystemExit(
            f"Vault artifacts already present in {cfg.vault_dir}. Delete this "
            "model's subdirectory first if you intend to re-stage (a stale "
            "manifest will fail verification against new files). Other models' "
            "vaults under base_dir are unaffected."
        )

    print(f"Model:   {MODEL_NAME}")
    print(f"Dtype:   {torch_dtype}")
    print(f"Vault:   {cfg.vault_dir}")
    print(f"\nLoading {MODEL_NAME} via TransformerLens (downloads from HF once)…")
    print("  NOTE: 6.9B in bf16 needs a HIGH-RAM runtime here (~25-30 GB peak).")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device="cpu", dtype=torch_dtype)

    # 1) Weights — TransformerLens-format state_dict (blocks.* keys), what
    #    HookedTransformer.load_state_dict(strict=True) expects in load_model().
    print(f"Writing weights → {weights_path}")
    torch.save(model.state_dict(), weights_path)

    # 2) Config — the dict HookedTransformerConfig.from_dict() consumes.
    #    model.cfg.to_dict() round-trips through from_dict() cleanly. The dtype
    #    serialises as e.g. "torch.bfloat16"; load_model() strips "torch." and
    #    maps it back, so the model inits in THIS dtype on load.
    print(f"Writing config → {config_path}")
    cfg_dict = model.cfg.to_dict()
    # JSON cannot hold dtype objects etc.; coerce non-serialisable values to str.
    config_path.write_text(json.dumps(cfg_dict, indent=2, default=str))

    # 3) Manifest — hashes computed AFTER writing, matching verify_manifest().
    #    Keys must be exactly the artifact names load_model() checks: weights, config.
    print(f"Writing manifest → {manifest_path}")
    manifest = {
        "weights": sha256(weights_path),
        "config":  sha256(config_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print("\nVault staged successfully:")
    for p in (weights_path, config_path, manifest_path):
        print(f"  {p}  ({p.stat().st_size:,} bytes)")
    print(f"\nweights sha256: {manifest['weights'][:16]}…")
    print(f"config  sha256: {manifest['config'][:16]}…")
    print(f"\nCell 2 load_model(VaultConfig(model_name='{MODEL_NAME}')) will now "
          "validate, verify, and load from this vault.")


if __name__ == "__main__":
    main()
