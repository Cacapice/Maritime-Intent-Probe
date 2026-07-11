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

PRECISION / MEMORY (revised 2026-07-11 — see CHANGELOG.md)
----------------------------------------------------------
6.9B is staged in bfloat16 (~13.8 GB on disk) rather than float32 (~27.6 GB).
That halves the Drive footprint, the CPU RAM needed here, and the VRAM needed in
Cell 2. vault.load_model() reads the dtype from config.json and inits the model
in that dtype, so the stage dtype IS the inference dtype — keep them consistent.
Override with VAULT_DTYPE ('bfloat16' | 'float16' | 'float32').

Because stage dtype = inference dtype, MATCH THE DTYPE TO THE INFERENCE GPU:
  • T4 / other pre-Ampere GPUs have no native bfloat16 — stage float16.
  • A100-class (Ampere+) — bfloat16 is fine and is the default.

Memory expectations (staging runs on CPU; peak is roughly 2–3× the weight
footprint while the HF and TransformerLens copies coexist):
  • 6.9B bf16: needs a HIGH-RAM runtime (~25–30 GB peak). The standard
    ~12.7 GB Colab runtime WILL be OOM-killed — use an A100/high-RAM runtime.
  • 1.4B: fp16 (~3 GB weights, ~8 GB peak) fits the standard runtime with
    headroom. bf16/fp32 staging of 1.4B is BORDERLINE on a standard runtime
    that has other allocations resident — observed OOM-killed in practice
    (2026-07-11). For 1.4B on a standard runtime, set VAULT_DTYPE=float16
    and run this script first, before anything else allocates RAM.
  • Loading in Cell 2 (GPU): 6.9B needs an A100-class (>=40 GB) runtime;
    1.4B fp16 runs on a T4 (~14 GB).

NOTE ON OOM: when Colab's OOM-killer terminates this script the only trace in
the notebook output is a bare '^C' after the tokenizer downloads. The pre-flight
check below turns that into an explicit error before the download starts.

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

The manifest is written LAST, so its presence is the completion marker: a vault
directory containing artifacts but no manifest is an interrupted stage, and this
script now removes those partial artifacts automatically before re-staging.
A vault WITH a manifest is never touched — delete that model's subdirectory
yourself if you intend to re-stage a completed vault.

After running this, Cell 2's load_model(CFG, ...) will validate → verify → load
cleanly.
"""
import os, json, hashlib

import torch
from transformer_lens import HookedTransformer

# Single source of truth for paths — never reconstruct them by hand here, or the
# manifest/layout can drift from what load_model() expects.
from config import VaultConfig

MODEL_NAME = os.getenv("VAULT_MODEL_NAME", "EleutherAI/pythia-6.9b")
DTYPE_NAME = os.getenv("VAULT_DTYPE", "bfloat16")
# Where from_pretrained materialises the model for processing. 'cuda' routes the
# weight-processing peak (which UPCASTS TO FP32 ON CPU — the actual cause of the
# standard-runtime OOM kills, regardless of VAULT_DTYPE) into GPU VRAM instead
# of CPU RAM. Weights are moved back to CPU before torch.save, so the vault
# format is identical either way.
DEVICE_NAME = os.getenv("VAULT_DEVICE", "cpu")

_DTYPE_MAP = {
    "float32":  torch.float32, "fp32": torch.float32,
    "float16":  torch.float16, "fp16": torch.float16,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
}

# Rough staging peaks (CPU RAM, GB) per model/dtype — deliberately conservative.
# Used only for the pre-flight warning; not a hard gate on non-Colab machines.
_PEAK_GB = {
    ("pythia-1.4b", torch.float16):  8,
    ("pythia-1.4b", torch.bfloat16): 12,
    ("pythia-1.4b", torch.float32):  14,
    ("pythia-6.9b", torch.float16):  28,
    ("pythia-6.9b", torch.bfloat16): 28,
    ("pythia-6.9b", torch.float32):  55,
}


def sha256(path, chunk_size: int = 1 << 20) -> str:
    """Identical to vault.sha256 — streaming SHA-256, 1 MiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _available_ram_gb() -> float | None:
    """MemAvailable from /proc/meminfo (Linux/Colab); None where unavailable."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 ** 2)   # kB → GB
    except OSError:
        pass
    return None


def _preflight(torch_dtype: torch.dtype) -> None:
    """Fail loudly BEFORE the download on predictable OOM; warn on dtype/GPU
    mismatches. Advisory only where the environment can't be inspected."""
    # 1) RAM check — converts the OOM-killer's silent '^C' into an explicit error.
    size_key = next((k for k in ("pythia-1.4b", "pythia-6.9b")
                     if k in MODEL_NAME.lower()), None)
    avail = _available_ram_gb()
    peak = _PEAK_GB.get((size_key, torch_dtype)) if size_key else None
    if avail is not None and peak is not None:
        print(f"Pre-flight: ~{avail:.1f} GB RAM available; "
              f"estimated staging peak ~{peak} GB ({MODEL_NAME}, {torch_dtype}).")
        if avail < peak:
            hint = (" Hint: VAULT_DTYPE=float16 halves the peak."
                    if torch_dtype != torch.float16 else
                    " Use a high-RAM runtime for this model.")
            raise SystemExit(
                f"ABORTING before download: available RAM ({avail:.1f} GB) is "
                f"below the estimated staging peak (~{peak} GB). Continuing "
                f"would end in a silent OOM kill ('^C').{hint}"
            )

    # 2) Stage dtype = inference dtype, so warn if the visible GPU can't run it.
    if torch_dtype is torch.bfloat16 and torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major < 8:   # pre-Ampere (e.g. T4 = 7.5): no native bf16
            print(
                "WARNING: staging in bfloat16, but the visible GPU "
                f"({torch.cuda.get_device_name(0)}) is pre-Ampere and has no "
                "native bf16 support. load_model() will init the model in the "
                "STAGED dtype — set VAULT_DTYPE=float16 unless inference will "
                "run on an Ampere+ GPU."
            )


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

    # The manifest is written last, so it is the completion marker.
    if manifest_path.exists():
        raise SystemExit(
            f"Completed vault already present in {cfg.vault_dir} (manifest "
            "found). Delete this model's subdirectory first if you intend to "
            "re-stage (a stale manifest will fail verification against new "
            "files). Other models' vaults under base_dir are unaffected."
        )
    partial = [p for p in (weights_path, config_path) if p.exists()]
    if partial:
        # No manifest + artifacts present = a prior stage was interrupted
        # (weights.pth is written first, so this is the common OOM leftover).
        print("Removing partial artifacts from an interrupted prior stage:")
        for p in partial:
            print(f"  {p}")
            p.unlink()

    device = DEVICE_NAME
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: VAULT_DEVICE=cuda but no GPU visible — falling back to CPU "
              "staging (higher RAM peak).")
        device = "cpu"

    if device == "cpu":
        _preflight(torch_dtype)
    else:
        print(f"Staging on {device}: processing peak lands in GPU VRAM; "
              "CPU pre-flight skipped.")

    print(f"Model:   {MODEL_NAME}")
    print(f"Dtype:   {torch_dtype}")
    print(f"Vault:   {cfg.vault_dir}")
    print(f"\nLoading {MODEL_NAME} via TransformerLens (downloads from HF once)…")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device, dtype=torch_dtype)

    # 1) Weights — TransformerLens-format state_dict (blocks.* keys), what
    #    HookedTransformer.load_state_dict(strict=True) expects in load_model().
    #    Moved to CPU first so the on-disk format is device-independent.
    print(f"Writing weights → {weights_path}")
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(state, weights_path)

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
    #    Written LAST: its presence marks the vault complete.
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
