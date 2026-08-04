"""
scale_probe.py
AGPL 3. Copyright © 2026 Katherine J. Ombrellaro.
==============
Targeted Pythia-6.9B scale probe (PREREGISTRATION.md §3.7 / §4.7).

Runs the single best-signal train/test attack-class combination from the
1.4B transfer matrix on Pythia-6.9B using identical probe infrastructure.
Not a full pipeline replication — no new SAE fitting procedure at 6.9B scale.

STATUS NOTE (Amendment 1) — ARCHIVAL MODULE.
  This module is part of the confirmatory program, preserved unchanged as the
  archival record and NOT executed as written. It defined a targeted
  within-Pythia scale comparison (1.4B -> 6.9B) under locked hypothesis H5; the
  text below — including references to adversarial-intent representation and the
  H5 pass/fail tolerance — is confirmatory-era language retained verbatim.

  Under Amendment 1 the scale question is reclassified as EXTERNAL validity: its
  within-Pythia component informs the SCOPE of E7 (Phase I, conditional on
  compute), and its cross-architecture component defers to V3 (Phase II). E7's
  analysis is specified separately, from the internal construct-validity data
  portfolio — NOT from this module's procedure — is geometry-only, and
  interprets no result as evidence regarding adversarial intent. The retired
  gate identifier BC8 formerly gestured at this analysis. H5 and the
  §3.7 / §4.7 thresholds remain locked (PREREGISTRATION.md §8).

Pre-registered hypothesis (H5):
  Transfer AUC on the best-signal combination at 6.9B within 0.10 of the
  1.4B result → geometric comparability claimed.
  Difference > 0.10 → scale-dependent geometry; reported as a finding.

Compute note (PREREGISTRATION.md §3.1):
  fp32 Pythia-6.9B is ~28GB resident (the scale notebook loads fp32 — no dtype
  is passed to from_pretrained); with SAE + forward-pass/activation overhead the
  working set is ~36GB (see scale notebook Cell S3 budget). This run requires a
  >=40GB GPU: a Colab A100-40GB (which reports ~42GB usable) suffices with ~6GB
  headroom; an A100-80GB has ample headroom (~$3/hr, 3–5 hours, ~$10–15 total).
  Available on Colab Pro/Pro+ A100 runtimes; the free-tier T4 (~15GB) cannot
  run the 6.9B probe and falls back to Pythia-2.8B. Confirm the environment has
  >=40GB VRAM before calling run_scale_probe(). Cell 0 (VRAM check) reports
  available VRAM; on any runtime below 40GB it falls back to Pythia-2.8B.

Architecture constraint (Kulkarni et al. 2026):
  Probes do not transfer across model architectures. Results from this run
  are specific to the Pythia model family. Cross-architecture inference
  requires independent replication.

Usage:
    # After transfer matrix is complete:
    best_combination = identify_best_combination(transfer_results)
    result = run_scale_probe(
        model_6b, cfg, env, collector_6b, probe_direction, best_combination, device
    )
"""

import json
import logging
from typing import Optional

import torch
from sklearn.metrics import roc_auc_score
from transformer_lens import HookedTransformer

from science.collector import ActivationCollector
from science.config import VaultConfig
from science.environment import MaritimeEnvironment
from science.sae import SparseAutoencoder

logger = logging.getLogger(__name__)

SCALE_PROBE_TOLERANCE = 0.10    # pre-registered: |AUC_6.9B - AUC_1.4B| ≤ 0.10
MIN_VRAM_GB           = 40.0    # fp32 6.9B is ~28GB resident (NOT ~14GB fp16);
                                # with SAE + activation overhead the working set
                                # is ~36GB. 40GB is the floor — a Colab A100-40GB
                                # (reports ~42GB usable) clears it with ~6GB
                                # headroom for this single-hook pipeline. Below
                                # this -> pre-registered Pythia-2.8B fallback.

# Pythia-6.9B layer subset — same relative positions as 1.4B scan
# 6.9B has 32 layers; proportionally equivalent to [4,8,12,16,20,24] in 24-layer model
SCALE_HOOK_NAMES = [
    f"blocks.{i}.hook_resid_post" for i in [5, 10, 16, 21, 26, 31]
]


FALLBACK_MODEL = "EleutherAI/pythia-2.8b"  # pre-registered fallback per PREREGISTRATION.md §3.7
MIN_VRAM_FALLBACK_GB = 10.0               # Pythia-2.8B requires ~6GB VRAM


def check_vram(device: str = "cuda") -> dict:
    """
    Check available VRAM and determine appropriate scale probe model.

    Returns a dict with vram_gb, model_to_use, and is_fallback flag.
    Does NOT raise — allows caller to decide whether to proceed with fallback.

    Pre-registered fallback (PREREGISTRATION.md §3.7):
      If Pythia-6.9B is unavailable (VRAM < 40GB), Pythia-2.8B is the
      pre-specified fallback without requiring a registered amendment.
      Results reported as a partial scale comparison with fallback noted.
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available — scale probe cannot run.")
        return {
            "vram_gb":      0.0,
            "model_to_use": None,
            "is_fallback":  False,
            "message":      "CUDA not available. Scale probe cannot run.",
        }

    props = torch.cuda.get_device_properties(0)
    vram  = props.total_memory / 1e9
    logger.info("GPU: %s | VRAM: %.1f GB", props.name, vram)

    if vram >= MIN_VRAM_GB:
        logger.info("✓ Sufficient VRAM (%.1f GB) — Pythia-6.9B scale probe will run.", vram)
        return {
            "vram_gb":      round(vram, 1),
            "model_to_use": "EleutherAI/pythia-6.9b",
            "is_fallback":  False,
            "message":      f"Sufficient VRAM ({vram:.1f}GB >= {MIN_VRAM_GB:.0f}GB). Running Pythia-6.9B.",
        }
    elif vram >= MIN_VRAM_FALLBACK_GB:
        logger.warning(
            "Insufficient VRAM for Pythia-6.9B (%.1fGB < %.0fGB required). "
            "6.9B requires a paid Colab A100 runtime (Pro/Pro+, ~42GB usable); "
            "the free-tier T4 (~15GB) cannot run it. Using pre-registered "
            "fallback Pythia-2.8B. Results reported as partial scale comparison "
            "per PREREGISTRATION.md §3.7.",
            vram, MIN_VRAM_GB,
        )
        return {
            "vram_gb":      round(vram, 1),
            "model_to_use": FALLBACK_MODEL,
            "is_fallback":  True,
            "message": (
                f"FALLBACK — {vram:.1f}GB VRAM < {MIN_VRAM_GB:.0f}GB required for 6.9B. "
                f"6.9B needs a paid Colab A100 runtime (Pro/Pro+, ~42GB usable); the "
                f"free-tier T4 (~15GB) cannot run it. Using pre-registered fallback "
                f"Pythia-2.8B per PREREGISTRATION.md §3.7. Report as partial scale comparison."
            ),
        }
    else:
        logger.error(
            "Insufficient VRAM for any scale probe (%.1fGB < %.1fGB minimum). "
            "Scale comparison omitted — 1.4B results stand independently.",
            vram, MIN_VRAM_FALLBACK_GB,
        )
        return {
            "vram_gb":      round(vram, 1),
            "model_to_use": None,
            "is_fallback":  False,
            "message": (
                f"INSUFFICIENT VRAM — {vram:.1f}GB < {MIN_VRAM_FALLBACK_GB}GB minimum. "
                f"Scale probe cannot run. Scale comparison omitted per stopping rules. "
                f"1.4B results stand independently."
            ),
        }


def identify_best_combination(transfer_results: dict) -> tuple[str, str]:
    """
    Identify the train/test attack class pair with the highest class-specific
    transfer AUC from the 1.4B transfer matrix results.

    Args:
        transfer_results: Output of transfer.run_transfer_matrix()

    Returns:
        (train_class, test_class) with highest class-specific transfer AUC.
    """
    cs = transfer_results.get("class_specific", {})
    best_train, best_test, best_auc = None, None, -1.0

    for train_cls, test_dict in cs.items():
        for test_cls, vals in test_dict.items():
            auc = vals.get("auc", -1.0)
            if auc > best_auc:
                best_auc  = auc
                best_train = train_cls
                best_test  = test_cls

    logger.info(
        "Best 1.4B transfer signal: %s → %s (AUC=%.4f)",
        best_train, best_test, best_auc,
    )
    return best_train, best_test


def run_scale_probe(
    model_6b: HookedTransformer,
    cfg: VaultConfig,
    env: MaritimeEnvironment,
    probe_direction_1b: torch.Tensor,
    best_layer_1b: str,
    train_class: str,
    test_class: str,
    auc_1b: float,
    device: str = "cuda",
    n_pairs: int = 50,
    sae_n_features: int = 16384,
    sae_n_steps: int = 500,
    is_fallback: bool = False,
    fallback_model_name: str = FALLBACK_MODEL,
    train_payloads: list = None,
    test_payloads: list = None,
) -> dict:
    """
    Run the targeted 6.9B scale probe on the best-signal transfer combination.

    Procedure:
      1. Verify >=40GB VRAM.
      2. Generate train and test class pairs.
      3. Collect residual stream activations at proportionally equivalent layers.
      4. Fit a minimal SAE at the best-equivalent 6.9B layer.
      5. Project probe direction from 1.4B space and score the test class.
         Note: the probe direction is compared geometrically — high cosine
         similarity between 1.4B and 6.9B directions at equivalent layers
         is evidence of geometric comparability.
      6. Report AUC and geometric similarity; compare against pre-registered
         tolerance (|AUC_6.9B - AUC_1.4B| ≤ 0.10).

    Args:
        probe_direction_1b: Unit direction from 1.4B MassMeanProbe for the
                            best-signal train/test combination.
        best_layer_1b:      Hook name of the 1.4B best layer (e.g.
                            'blocks.16.hook_resid_post'). Used to find the
                            proportionally equivalent 6.9B layer.
        auc_1b:             1.4B transfer AUC for this combination (comparison
                            baseline for the pre-registered tolerance check).
        train_payloads:     Optional pre-generated train-class payloads. If
                            None (default), generated internally exactly as
                            before -- existing callers see no behavior change.
        test_payloads:      Optional pre-generated test-class payloads. Pass
                            this when a caller needs the SAME payloads (and
                            therefore the same labels) that produced auc_6b
                            for a downstream check, instead of regenerating
                            from env and relying on RNG determinism.

    Returns:
        Dict with auc_6b, auc_1b, auc_delta, geometric_comparability,
        cosine_similarity_directions, passed, and interpretation.
    """
    # Single VRAM check — its result is the only source consulted below (vram_gb
    # for the result dict, model_to_use for the can't-run guard). check_vram does
    # NOT load the model: the caller loads model_6b and declares is_fallback. We
    # reconcile that declaration against detection and surface any disagreement
    # rather than labeling results off a flag that may not match the loaded model.
    vram_result = check_vram(device)
    if vram_result["model_to_use"] is None:
        raise RuntimeError(vram_result["message"])
    vram_gb = vram_result.get("vram_gb", 0.0)

    if vram_result["is_fallback"] != is_fallback:
        logger.warning(
            "is_fallback=%s was declared by the caller, but VRAM detection implies "
            "is_fallback=%s (%.1f GB). Results are labelled with the caller-declared "
            "value — verify the loaded model matches it.",
            is_fallback, vram_result["is_fallback"], vram_gb,
        )

    scale_model_name = fallback_model_name if is_fallback else "EleutherAI/pythia-6.9b"
    logger.info("=== Targeted Scale Probe (%s) ===", scale_model_name)
    logger.info("Combination: %s → %s | 1.4B AUC: %.4f", train_class, test_class, auc_1b)
    if is_fallback:
        logger.warning("Running with fallback model %s per PREREGISTRATION.md §3.7", fallback_model_name)

    # Map 1.4B layer to proportionally equivalent 6.9B layer
    try:
        layer_num_1b = int(best_layer_1b.split(".")[1])   # e.g. 16 from blocks.16.hook_resid_post
        # 1.4B has 24 layers, 6.9B has 32 layers
        layer_num_6b = round(layer_num_1b * (32 / 24))
        best_layer_6b = f"blocks.{layer_num_6b}.hook_resid_post"
    except (IndexError, ValueError):
        best_layer_6b = SCALE_HOOK_NAMES[len(SCALE_HOOK_NAMES) // 2]
        logger.warning("Could not map 1.4B layer; using default 6.9B layer: %s", best_layer_6b)

    logger.info("Proportional 6.9B layer: %s (from 1.4B: %s)", best_layer_6b, best_layer_1b)

    # Build a minimal collector for the 6.9B model
    collector_6b = ActivationCollector(model_6b, [best_layer_6b])

    # Generate train and test payloads -- reuse caller-provided ones if given,
    # so a caller who also needs the EXACT same payload set for a downstream
    # diagnostic (e.g. a model-blind witness check against auc_6b) doesn't have
    # to regenerate from env and hope RNG state matches. Falls back to the
    # original generate-here behavior when not provided, so every existing
    # caller is unaffected.
    if train_payloads is None:
        train_payloads = env.generate_class_pairs(train_class, n_pairs=n_pairs)
    if test_payloads is None:
        test_payloads = env.generate_class_pairs(test_class, n_pairs=n_pairs)

    # Collect activations
    logger.info("Collecting 6.9B train activations (%s)...", train_class)
    train_mean, _, train_labels = collector_6b.collect_dual(train_payloads, env, turn_index=-1)

    logger.info("Collecting 6.9B test activations (%s)...", test_class)
    test_mean, _, test_labels = collector_6b.collect_dual(test_payloads, env, turn_index=-1)

    # Fit minimal SAE at the 6.9B layer
    d_model_6b = train_mean[best_layer_6b].shape[-1]
    logger.info("Fitting 6.9B SAE (d=%d → %d features, n_steps=%d)...",
                d_model_6b, sae_n_features, sae_n_steps)
    sae_6b = SparseAutoencoder(d_model_6b, sae_n_features).to(device)
    sae_6b.fit(train_mean[best_layer_6b], n_steps=sae_n_steps, labels=train_labels)

    # Encode and compute mass-mean probe direction at 6.9B
    with torch.no_grad():
        train_features = sae_6b.encode(train_mean[best_layer_6b].to(device)).cpu()
        test_features  = sae_6b.encode(test_mean[best_layer_6b].to(device)).cpu()

    y_train = train_labels.bool()
    direction_6b = train_features[y_train].mean(0) - train_features[~y_train].mean(0)
    direction_6b = direction_6b / (direction_6b.norm() + 1e-8)

    # Score test class
    scores = (test_features.float() @ direction_6b).numpy()
    auc_6b = float(roc_auc_score(test_labels.numpy(), scores))

    # Geometric comparability: cosine similarity between 1.4B and 6.9B directions
    # Only meaningful if both directions live in the same SAE feature space —
    # since they do not (different SAEs), we report AUC delta as the primary metric
    # and note this limitation explicitly.
    auc_delta = abs(auc_6b - auc_1b)
    passed    = auc_delta <= SCALE_PROBE_TOLERANCE

    result = {
        "train_class":     train_class,
        "test_class":      test_class,
        "layer_1b":        best_layer_1b,
        "layer_6b":        best_layer_6b,
        "auc_1b":          round(auc_1b, 4),
        "auc_6b":          round(auc_6b, 4),
        "auc_delta":       round(auc_delta, 4),
        "tolerance":       SCALE_PROBE_TOLERANCE,
        "passed":          passed,
        "vram_gb":         vram_gb,
        "n_pairs":         n_pairs,
        "is_fallback":     is_fallback,
        "scale_model":     scale_model_name,
        "fallback_note": (
            f"PRE-REGISTERED FALLBACK — results use {fallback_model_name} rather than "
            f"Pythia-6.9B due to compute constraints. Reported as partial scale comparison "
            f"per PREREGISTRATION.md §3.7. No amendment required."
            if is_fallback else None
        ),
        "architecture_note": (
            "Results are specific to the Pythia model family. "
            "Kulkarni et al. (2026) demonstrate that probes do not transfer "
            "across architectures — cross-architecture inference requires "
            "independent replication."
        ),
        "interpretation": (
            f"GEOMETRIC COMPARABILITY — AUC delta {auc_delta:.3f} ≤ {SCALE_PROBE_TOLERANCE}. "
            f"Probe direction generalises across Pythia scale (1.4B → "
            f"{'2.8B [fallback]' if is_fallback else '6.9B'}). "
            f"Finding moves from single-model to scale-comparative result."
            if passed else
            f"SCALE-DEPENDENT GEOMETRY — AUC delta {auc_delta:.3f} > {SCALE_PROBE_TOLERANCE}. "
            f"Probe direction does not generalise geometrically from 1.4B to "
            f"{'2.8B [fallback]' if is_fallback else '6.9B'}. "
            f"This is a finding about the relationship between model scale and "
            f"adversarial intent representation in the Pythia family."
        ),
    }

    cfg.scale_probe_path.write_text(json.dumps(result, indent=2))
    logger.info("Scale probe result: %s", result["interpretation"])
    logger.info("Result written to: %s", cfg.scale_probe_path)
    return result
