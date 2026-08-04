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

WEIGHT PROCESSING (VAULT_PROCESSING, 2026-07-17)
------------------------------------------------
    VAULT_PROCESSING=none      python stage_vault.py   # default; raw HF weights
    VAULT_PROCESSING=standard  python stage_vault.py   # TL interpretability prep

"none" stages via from_pretrained_no_processing; "standard" via from_pretrained.
These produce DIFFERENT MODELS -- fold_ln rewrites LayerNorm into the following
linear layer, so hook_resid_post differs -- and results across the two are not
comparable. The mode is written to manifest.json under `_processing`, and
vault.load_model(..., expect_processing=...) refuses a mismatch. The default is
"none" because that is what the notebook's own model load uses.

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

The manifest is written LAST, so its presence marks completion — but only when
the artifacts it vouches for are still there. Two failure states are handled
automatically:

  artifacts, no manifest  -> interrupted stage (the common OOM leftover); the
                             partial artifacts are removed and staging proceeds.
  manifest, no artifacts  -> BROKEN vault. The weights disappeared AFTER a
                             successful stage — on Colab this is almost always a
                             Drive write that never synced or was rejected on
                             quota, since a 6.9B bf16 weights.pth is ~13.8 GB.
                             Previously this deadlocked the pipeline: stage_vault
                             refused to re-stage (manifest found) while
                             cfg.validate() refused to load (weights missing).
                             The broken vault is now cleared and re-staged, with
                             the missing artifacts named.

A COMPLETE vault (manifest + all artifacts) is still never touched — delete that
model's subdirectory yourself if you intend to re-stage it.

After running this, Cell 2's load_model(CFG, ...) will validate → verify → load
cleanly.
"""
import os, sys, json, hashlib

import torch
from transformer_lens import HookedTransformer

# Single source of truth for paths — never reconstruct them by hand here, or the
# manifest/layout can drift from what load_model() expects.
from science.config import VaultConfig

MODEL_NAME = os.getenv("VAULT_MODEL_NAME", "EleutherAI/pythia-6.9b")
DTYPE_NAME = os.getenv("VAULT_DTYPE", "bfloat16")
# Where from_pretrained materialises the model for processing. 'cuda' routes the
# weight-processing peak (which UPCASTS TO FP32 ON CPU — the actual cause of the
# standard-runtime OOM kills, regardless of VAULT_DTYPE) into GPU VRAM instead
# of CPU RAM. Weights are moved back to CPU before torch.save, so the vault
# format is identical either way.
DEVICE_NAME = os.getenv("VAULT_DEVICE", "cpu")

# ── WEIGHT PROCESSING (VAULT_PROCESSING) ──────────────────────────────────────
# TransformerLens can rewrite the weights on load. from_pretrained() does so by
# DEFAULT; from_pretrained_no_processing() is the same function with those flags
# off. The difference is not cosmetic: fold_ln rewrites each LayerNorm's learned
# scale into the following linear layer, and center_writing_weights shifts the
# directions that write into the residual stream -- so blocks.N.hook_resid_post
# does NOT hold the same vector under the two modes. Every probe direction, every
# Δ, every AUC differs.
#
# This vault previously staged with from_pretrained (processing ON) while the
# notebook loaded via from_pretrained_no_processing (processing OFF): two
# different models, no error, silently incomparable numbers. The mode is now
# EXPLICIT, defaults to the notebook's ("none"), and is RECORDED IN THE MANIFEST
# so vault.load_model can refuse a vault staged in the other mode.
#
#   VAULT_PROCESSING=none      (default) raw HF weights -- matches the notebook's
#                              from_pretrained_no_processing. Defensible for this
#                              study precisely because folding changes what
#                              resid_post means, and the claims are about
#                              displacement in that stream.
#   VAULT_PROCESSING=standard  TransformerLens's conventional interpretability
#                              preprocessing (fold_ln, center_writing_weights,
#                              center_unembed, fold_value_biases).
#
# Changing this changes the MODEL the experiment runs on. It is not a convenience
# flag: results across modes are not comparable, and switching it invalidates any
# artifact accumulated under the other mode.
PROCESSING_MODE = os.getenv("VAULT_PROCESSING", "none").strip().lower()

_PROCESSING_FLAGS = {
    "none":     dict(fold_ln=False, center_writing_weights=False,
                     center_unembed=False, fold_value_biases=False),
    "standard": dict(fold_ln=True, center_writing_weights=True,
                     center_unembed=True, fold_value_biases=True),
}

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


_USAGE = f"""stage_vault.py — one-time creation of the integrity-verified model vault.

    python stage_vault.py [--dry-run] [--help]

Configured entirely by environment variables (read at MODULE level, so they must
be set BEFORE the process starts — which is why the notebook cell runs this as a
subprocess with an explicit env rather than importing it):

    VAULT_MODEL_NAME   {MODEL_NAME}
                       the model to stage; also selects the per-model vault dir.
    VAULT_DTYPE        {DTYPE_NAME}
                       float32 | float16 | bfloat16. STAGE DTYPE = INFERENCE
                       DTYPE — load_model() inits the model in whatever was
                       staged, so match it to the inference GPU (T4 has no native
                       bf16; use float16).
    VAULT_DEVICE       {DEVICE_NAME}
                       cpu | cuda. 'cuda' routes the weight-processing peak
                       (which upcasts to fp32) into VRAM instead of CPU RAM —
                       this is what prevents the silent OOM kill on a standard
                       runtime.
    VAULT_PROCESSING   {PROCESSING_MODE}
                       none | standard. WHICH MODEL is staged, not how it is
                       stored. 'none' = from_pretrained_no_processing (raw HF
                       weights, matches the notebook); 'standard' = from_pretrained
                       (fold_ln etc). Results across modes are NOT comparable.
    VAULT_BASE_DIR     (see config.py)

Flags:
    --dry-run   print the plan and run the RAM pre-flight, then exit WITHOUT
                downloading. Worth doing before committing to a ~14 GB pull.
    --help      this message.
"""


def _dry_run(cfg, torch_dtype) -> None:
    """Report exactly what a real run would do, without the download."""
    flags = _PROCESSING_FLAGS[PROCESSING_MODE]
    print(_USAGE.split("Flags:")[0])
    print("── plan ─────────────────────────────────────────────────────────────")
    print(f"  model        {MODEL_NAME}")
    print(f"  dtype        {torch_dtype}")
    print(f"  device       {DEVICE_NAME}")
    print(f"  processing   {PROCESSING_MODE}  ->  "
          f"{'from_pretrained_no_processing' if PROCESSING_MODE == 'none' else 'from_pretrained'}")
    print(f"               {flags}")
    print(f"  vault dir    {cfg.vault_dir}")
    for name, p in (("weights", cfg.weights_path), ("config", cfg.config_path),
                    ("manifest", cfg.manifest_path)):
        state = "EXISTS" if p.exists() else "will be written"
        print(f"    {name:<9} {p}  [{state}]")

    if cfg.manifest_path.exists():
        missing = [n for n, p in cfg.artifacts.items() if not p.exists()]
        if missing:
            print(f"\n  BROKEN VAULT: manifest present but {missing} missing — a real "
                  "run would clear and re-stage it.")
        else:
            print("\n  COMPLETE VAULT: a real run would REFUSE (delete the model's "
                  "subdirectory first to re-stage).")

    print("\n── pre-flight ───────────────────────────────────────────────────────")
    if DEVICE_NAME.startswith("cuda"):
        print("  Staging on cuda: the processing peak lands in VRAM; CPU RAM "
              "pre-flight does not apply.")
    else:
        try:
            _preflight(torch_dtype)
            print("  CPU RAM pre-flight: OK")
        except SystemExit as e:
            print(f"  CPU RAM pre-flight WOULD ABORT:\n    {e}")
    print("\nDry run only — nothing downloaded, nothing written.")


def main():
    if "--help" in sys.argv or "-h" in sys.argv:
        print(_USAGE)
        return
    if PROCESSING_MODE not in _PROCESSING_FLAGS:
        raise SystemExit(
            f"Unknown VAULT_PROCESSING='{PROCESSING_MODE}'. Valid: "
            f"{sorted(_PROCESSING_FLAGS)}. This selects WHICH MODEL is staged, "
            "not how it is stored -- see the module docstring."
        )
    if DTYPE_NAME.replace("torch.", "").lower() not in _DTYPE_MAP:
        raise SystemExit(f"Unknown VAULT_DTYPE='{DTYPE_NAME}'. Valid: {sorted(_DTYPE_MAP)}")
    torch_dtype = _DTYPE_MAP[DTYPE_NAME.replace("torch.", "").lower()]

    # Paths come from the same config the notebook/load_model use, keyed to this
    # model. checkpoint_version is None here, so weights/config/manifest land flat
    # in the per-model vault dir (base_dir / model_slug).
    cfg = VaultConfig(model_name=MODEL_NAME)

    if "--dry-run" in sys.argv:
        _dry_run(cfg, torch_dtype)
        return

    cfg.vault_dir.mkdir(parents=True, exist_ok=True)

    weights_path  = cfg.weights_path
    config_path   = cfg.config_path
    manifest_path = cfg.manifest_path

    # The manifest is written last, so it is the completion marker.
    # A manifest marks completion ONLY if the artifacts it vouches for still
    # exist. Treating its mere presence as "complete" deadlocks a vault whose
    # weights vanished AFTER staging (Drive sync failure, quota rejection, a
    # manual clean-up): stage_vault refuses to re-stage because the manifest is
    # there, while cfg.validate() refuses to load because the weights are not.
    # Nothing can then proceed without manual intervention, and neither message
    # tells you that is the situation you are in.
    if manifest_path.exists():
        vouched = {name: p for name, p in cfg.artifacts.items()}
        absent = {name: p for name, p in vouched.items() if not p.exists()}
        if not absent:
            raise SystemExit(
                f"Completed vault already present in {cfg.vault_dir} (manifest "
                "found, all artifacts present). Delete this model's subdirectory "
                "first if you intend to re-stage (a stale manifest will fail "
                "verification against new files). Other models' vaults under "
                "base_dir are unaffected."
            )
        # Manifest present, artifacts missing -> BROKEN, not complete.
        print(
            "BROKEN VAULT DETECTED — manifest present but the artifacts it "
            "vouches for are gone:"
        )
        for name, p in absent.items():
            print(f"  MISSING  {name}: {p}")
        for name, p in vouched.items():
            if p.exists():
                print(f"  present  {name}: {p}  ({p.stat().st_size:,} bytes)")
        print(
            "\nThis is the state that deadlocks the pipeline: the manifest says "
            "'complete', validate() says 'incomplete', and the old completion "
            "check refused to re-stage. A large weights file disappearing after a "
            "successful stage usually means the Drive write never synced or was "
            "rejected on quota -- check Drive free space before re-staging, or "
            "the same thing will happen again.\n"
            "Clearing the broken vault and re-staging:"
        )
        for name, p in vouched.items():
            if p.exists():
                print(f"  removing {p}")
                p.unlink()
        print(f"  removing {manifest_path}")
        manifest_path.unlink()

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

    flags = _PROCESSING_FLAGS[PROCESSING_MODE]
    print(f"Model:      {MODEL_NAME}")
    print(f"Dtype:      {torch_dtype}")
    print(f"Processing: {PROCESSING_MODE}  {flags}")
    print(f"Vault:      {cfg.vault_dir}")
    print(f"\nLoading {MODEL_NAME} via TransformerLens (downloads from HF once)…")
    if PROCESSING_MODE == "none":
        # The EXACT call the notebook makes, not an equivalent one. It is a thin
        # wrapper over from_pretrained with the four flags False, so calling it
        # guarantees the vault holds what the notebook would itself have built.
        model = HookedTransformer.from_pretrained_no_processing(
            MODEL_NAME, device=device, dtype=torch_dtype
        )
    else:
        model = HookedTransformer.from_pretrained(
            MODEL_NAME, device=device, dtype=torch_dtype, **flags
        )

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
    # `_processing` is METADATA, not an artifact hash. verify_manifest() iterates
    # cfg.artifacts (weights, config) and ignores every other key, so adding this
    # cannot alter integrity checking. The underscore prefix matches the
    # _summary/_structural convention used elsewhere in this repo.
    manifest = {
        "weights": sha256(weights_path),
        "config":  sha256(config_path),
        "_processing": {
            "mode": PROCESSING_MODE,
            **flags,
            "normalization_type": cfg_dict.get("normalization_type"),
            "staged_call": ("from_pretrained_no_processing" if PROCESSING_MODE == "none"
                            else "from_pretrained"),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print("\nVault staged successfully:")
    for p in (weights_path, config_path, manifest_path):
        print(f"  {p}  ({p.stat().st_size:,} bytes)")
    print(f"\nweights sha256: {manifest['weights'][:16]}…")
    print(f"config  sha256: {manifest['config'][:16]}…")
    print(f"\nProcessing mode recorded in manifest: {PROCESSING_MODE} "
          f"(normalization_type={cfg_dict.get('normalization_type')})")
    print(f"Cell 2 load_model(VaultConfig(model_name='{MODEL_NAME}'), "
          f"expect_processing='{PROCESSING_MODE}') will now validate, verify, and "
          "load from this vault. Passing expect_processing is what turns a mode "
          "mismatch into an error instead of a silent divergence.")


if __name__ == "__main__":
    main()
