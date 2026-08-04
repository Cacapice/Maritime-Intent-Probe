"""
experiment.py
AGPL 3. Copyright © 2026 Katherine J. Ombrellaro.
=============
ProbeExperiment: full experimental pipeline orchestrator.

Pipeline:
  generate pairs → collect activations per turn → fit SAE per layer
  → mass-mean probe + logistic regression probe in SAE feature space
  → accumulation curve + layer heatmap
  → causal comparison (selects the empirically more causal probe direction)
  → W&B logging + JSONL export

The causal comparison (compare_probes) runs directional patching with both
probe directions and selects the one with the higher causal rate as
best_probe_method (with an inconclusive band rather than a significance test of
the Marks & Tegmark (2023) claim). This selection is then used in
cross-attack-class and cross-domain transfer analyses, so subsequent results
rest on an empirically validated method choice rather than an assumed one.

RunMode controls all derived parameters in a single flag:
  RunMode.ITERATION — fast scan (n_steps=500, n_pairs=50, n_features=8192)
  RunMode.FINAL     — full run  (n_steps=2000, n_pairs=200, n_features=16384)
"""

import json
import logging
import re
import hashlib
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import numpy as np
import torch
import wandb

from science import collector
from science.collector import ActivationCollector
from science import config
from science.config import VaultConfig
from science import environment
from science.environment import MaritimeEnvironment, RoutingPayload
from science import patching
from science.patching import CausalPatcher
from science import probe
from science.probe import MassMeanProbe, LogisticRegressionProbe, ProbeResult, compare_probes
from science import sae
from science.sae import SparseAutoencoder
# CI helpers live in stats.py (single source of truth), aliased to the prior
# private names so call sites are unchanged. The stats versions round to 4 dp
# internally; this is immaterial here because every call site already rounds to
# 4 dp and the persisted JSON is byte-identical (only sub-0.1% log-display
# digits could differ in principle).
from science.stats import wilson_ci as _wilson_ci, normal_ci as _normal_ci

logger = logging.getLogger(__name__)


# ── Run mode ──────────────────────────────────────────────────────────────────

class RunMode(Enum):
    """
    Single flag controlling all derived experiment parameters.

    Prevents the most common configuration error: running a final-mode SAE
    with iteration-mode pairs or vice versa.

    ITERATION — fast pipeline scan to confirm signal exists before committing
                to a full run. Use during pre-month-1 preparation and month 1.
                n_steps=500, n_pairs=50, sae_n_features=8192
                Expected runtime: 30–45 min on A100, 2–3h on T4.

    FINAL     — full experimental run for results. Use in months 2 and 3
                once iteration runs have confirmed signal and null control
                has passed.
                n_steps=2000, n_pairs=200, sae_n_features=16384
                Expected runtime: 1–2h on A100, 6–8h on T4.
    """
    ITERATION = "iteration"
    FINAL     = "final"


# Parameter presets keyed by RunMode
_MODE_PARAMS: dict[RunMode, dict] = {
    RunMode.ITERATION: {
        "sae_n_steps":    500,
        "n_pairs":        50,
        "sae_n_features": 8192,
    },
    RunMode.FINAL: {
        "sae_n_steps":    2000,
        "n_pairs":        200,
        "sae_n_features": 16384,
    },
}


@runtime_checkable
class EnvironmentProtocol(Protocol):
    """Interface ProbeExperiment expects from an environment (documentation/typing
    only; not enforced at runtime beyond isinstance if ever used). A follow-on
    data-backed environment (e.g. real maritime-lane-weighted stimuli) is a drop-in
    as long as it satisfies this contract — experiment.py needs no change.

    Required:
        generate_pairs(n_pairs) -> list[RoutingPayload]
        reset_rng(seed=None) -> None          # deterministic dataset regeneration
        tokenize(payload) -> list[Tensor]
    Optional (read inertly if present):
        dataset_provenance -> dict            # recorded in the run manifest
    """
    def generate_pairs(self, n_pairs: int = ...): ...
    def reset_rng(self, seed=...) -> None: ...
    def tokenize(self, payload): ...


class ProbeExperiment:
    """
    Full experimental pipeline. See module docstring for pipeline overview.

    Args:
        model:          Loaded HookedTransformer.
        cfg:            VaultConfig — used for run name and export paths.
        environment:            MaritimeEnvironment (or any compatible environment).
        collector:      ActivationCollector configured with the target layer subset.
        probe:          MassMeanProbe instance (shared across runs for transfer tests).
        patching:        CausalPatcher for causal confirmation and comparison.
        mode:           RunMode.ITERATION or RunMode.FINAL. Sets n_steps, n_pairs,
                        and sae_n_features automatically. Pass explicit overrides
                        only if you need to deviate from the presets.
        sae_n_features: Override SAE dictionary size (default: from mode preset).
        sae_l1_coeff:   Sparsity penalty (tune to ~1–5% feature density).
        sae_n_steps:    Override SAE training steps (default: from mode preset).
        max_turns:      Number of turns to evaluate in the accumulation curve.
    """

    def __init__(
        self,
        model,
        cfg: VaultConfig,
        environment: MaritimeEnvironment,
        collector: ActivationCollector,
        probe: MassMeanProbe,
        patching: CausalPatcher,
        mode: RunMode = RunMode.ITERATION,
        sae_n_features: Optional[int] = None,
        sae_l1_coeff: float = 2e-4,
        sae_n_steps: Optional[int] = None,
        max_turns: int = 3,
        cv_sae_n_steps: Optional[int] = None,
        lr_probe_C: float = 0.1,
    ):
        params = _MODE_PARAMS[mode]

        self.model          = model
        self.cfg            = cfg
        self.environment    = environment
        self.collector      = collector
        self.probe          = probe
        # lr_probe_C: LogisticRegressionProbe's L2 regularisation strength.
        # Defaults to 0.1 (10x stronger penalty than sklearn's C=1.0 default) —
        # see LogisticRegressionProbe's docstring for the small-n/high-p
        # rationale. Exposed here, rather than hardcoded inside
        # LogisticRegressionProbe alone, so the SAME value is visibly threaded
        # through every probe instance this experiment creates (self.lr_probe,
        # self.raw_lr_probe) and therefore applied identically across every
        # layer, attack class, and transfer-matrix cell (H2-H5). If tuned via
        # select_C_nested_cv (probe.py), set the selected value here once
        # before running — do not vary it per layer/class/combination.
        self.lr_probe_C     = lr_probe_C
        self.lr_probe       = LogisticRegressionProbe(C=lr_probe_C)
        # Raw-space probe instances — separate from the SAE-space probes so that
        # stored directions never overwrite the SAE-space directions used by
        # compare_probes() and patching. These receive sae=None and sae_factory=None
        # so evaluate() operates directly on raw residual activations via the
        # existing sae=None branch in MassMeanProbe/LogisticRegressionProbe.
        self.raw_probe    = MassMeanProbe()
        self.raw_lr_probe = LogisticRegressionProbe(C=lr_probe_C)
        self.patching       = patching
        self.mode           = mode
        self.sae_n_features = sae_n_features if sae_n_features is not None else params["sae_n_features"]
        self.sae_l1_coeff   = sae_l1_coeff
        self.sae_n_steps    = sae_n_steps    if sae_n_steps    is not None else params["sae_n_steps"]
        # Steps for the per-fold SAEs used in leakage-free CV. Defaults to the
        # main step count for a faithful estimate; lower it to trade fidelity
        # for speed, since per-fold fitting multiplies SAE training by n_splits.
        self.cv_sae_n_steps = cv_sae_n_steps if cv_sae_n_steps is not None else self.sae_n_steps
        self._n_pairs       = params["n_pairs"]   # used as default in run()
        self.max_turns      = max_turns
        self._saes: dict[str, SparseAutoencoder] = {}
        self._bias_diagnostics: dict = {}
        # Per-turn LogisticRegressionProbe results, retained for the
        # supplementary AUC-by-depth diagnostic only (mass-mean drives selection).
        self._lr_results_by_turn: dict = {}
        # Per-turn raw-space probe results — parallel to the SAE-space results.
        # Used only for the direction cosine similarity diagnostic and the
        # raw-space AUC-by-depth heatmap; selection is unchanged.
        self._raw_results_by_turn:    dict = {}
        self._raw_lr_results_by_turn: dict = {}

        logger.info(
            "ProbeExperiment | mode=%s | n_pairs=%d | sae_n_steps=%d | sae_n_features=%d",
            mode.value, self._n_pairs, self.sae_n_steps, self.sae_n_features,
        )

    @staticmethod
    def _layer_seed(base_seed: int, layer: str) -> int:
        """Derive a stable per-layer seed from the run's base seed, so SAE
        initialization/training is reproducible (FIX 2026-06-20: previously
        unseeded -- see CHANGELOG.md) while still giving each layer a distinct
        seed rather than identical initialization across layers. Stable across
        Python process restarts (unlike hash(), which is salted per-process)."""
        digest = hashlib.sha256(f"{base_seed}:{layer}".encode()).hexdigest()
        return int(digest[:8], 16)

    def _get_or_fit_sae(
        self,
        layer: str,
        activations: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> SparseAutoencoder:
        """
        Fit a fresh SAE for this layer if not already cached.
        Passes labels for per-class reconstruction error logging (Bias Control 4).
        """
        if layer not in self._saes:
            d_model = activations.shape[-1]
            device  = next(self.model.parameters()).device
            base_seed = getattr(self.cfg, "seed", None)
            if base_seed is not None:
                torch.manual_seed(self._layer_seed(base_seed, layer))
            else:
                logger.warning(
                    "No cfg.seed set -- SAE fit for layer %s is NOT reproducible "
                    "across runs. Set cfg.seed for deterministic SAE-space results.",
                    layer,
                )
            sae     = SparseAutoencoder(d_model, self.sae_n_features, self.sae_l1_coeff).to(device)
            logger.info("Fitting SAE for layer %s (d=%d → %d features, device=%s)...",
                        layer, d_model, self.sae_n_features, device)
            fit_result = sae.fit(activations, n_steps=self.sae_n_steps, labels=labels)
            # Store reconstruction diagnostics for bias reporting
            if "per_class_mse" not in self._bias_diagnostics:
                self._bias_diagnostics["per_class_mse"] = {}
            self._bias_diagnostics["per_class_mse"][layer] = fit_result.get("per_class_mse")
            self._saes[layer] = sae
        return self._saes[layer]

    def _make_sae_factory(self):
        """Returns a closure (X_train_raw_np) -> frozen SAE, fitting a fresh SAE
        on the supplied raw TRAINING activations. Passed to probe.evaluate so the
        cross-validated AUC is computed with the SAE fit inside each fold (no
        leakage from the held-out activations into the dictionary). Hyperparameters
        are locked to the experiment's; steps follow cv_sae_n_steps.

        FIX (2026-06-20): seeded via cfg.seed (see CHANGELOG.md). Every fold
        uses the SAME seed for SAE init/training -- deliberate, not an
        oversight: this isolates the effect of the held-out data across folds
        from confounding random-init variance, and makes cross-validated AUC
        reproducible run-to-run. Previously unseeded.
        """
        device = next(self.model.parameters()).device
        n_feat, l1, steps = self.sae_n_features, self.sae_l1_coeff, self.cv_sae_n_steps
        base_seed = getattr(self.cfg, "seed", None)
        if base_seed is None:
            logger.warning(
                "No cfg.seed set -- cross-validated SAE fits are NOT reproducible "
                "across runs. Set cfg.seed for deterministic SAE-space CV results."
            )

        def factory(X_train_np):
            d_model = X_train_np.shape[-1]
            if base_seed is not None:
                torch.manual_seed(base_seed)
            sae = SparseAutoencoder(d_model, n_feat, l1).to(device)
            sae.fit(torch.tensor(X_train_np, dtype=torch.float32).to(device), n_steps=steps)
            sae.eval()                       # freeze before it is used to encode
            return sae

        return factory

    def run(self, n_pairs: Optional[int] = None) -> dict:
        """
        Run the full pipeline and return a results report dict.

        Args:
            n_pairs: Number of matched pairs. Defaults to the mode preset
                     (50 for ITERATION, 200 for FINAL). Pass explicitly only
                     to override the preset.
        """
        n = n_pairs if n_pairs is not None else self._n_pairs
        # Model-variation guard: drop any requested scan layer that does not exist
        # in this model (e.g. the registered set's top index on a shallower model).
        # Recorded in the run manifest so the layers actually scanned are auditable.
        self._layer_validation = self._validate_scan_layers()
        # Reproducible dataset: reset the env RNG so this run's pairs are
        # deterministic and refine_layers()/re-estimates draw the same set,
        # regardless of any RNG consumption since the env was constructed.
        self.environment.reset_rng()
        logger.info("Generating %d matched pairs (mode=%s)...", n, self.mode.value)
        payloads        = self.environment.generate_pairs(n)
        # FRAME-LEVEL CV GROUPS. Row-aligned with the activation tensors, since
        # collect_dual() consumes `payloads` in this exact order. Payloads inside
        # one template frame differ ONLY in slot fill (imo/dest/cargo), so
        # example-level folds train on one realisation and score its
        # near-duplicate siblings; the effective sample size is the number of
        # FRAMES, not payloads (semantic: 7 frames behind 50 payloads). Passed to
        # every evaluate() below so both the probe folds and the per-fold SAE fit
        # respect frame boundaries. See probe._make_splitter.
        groups          = [p.template_key for p in payloads]
        results_by_turn = {}
        sae_factory     = self._make_sae_factory()

        # ── Turn accumulation curve ───────────────────────────────────────────
        for turn_idx in range(self.max_turns):
            logger.info("Collecting activations: turn %d (dual collection)...", turn_idx)
            # Dual collection — both mean-pooled and final-token (Bias Control 3)
            mean_acts, final_acts, labels = self.collector.collect_dual(
                payloads, self.environment, turn_index=turn_idx
            )

            # Fit the DEPLOYABLE SAE per layer (cached after the first turn) on
            # the full set — used for the stored direction, in-sample AUC, and
            # downstream causal/transfer work. The reported held-out AUC is made
            # leakage-free separately via sae_factory (fresh SAE per CV fold).
            for layer, X in mean_acts.items():
                self._get_or_fit_sae(layer, X, labels=labels)  # passes labels for MSE logging

            # Run both probes per layer. sae = deployable (direction + in-sample);
            # sae_factory = per-fold SAE for the leakage-free cross-validated AUC.
            mm_results: list = []
            lr_results: list = []
            for layer, X in mean_acts.items():
                sae = self._saes[layer]
                mm_results += self.probe.evaluate(
                    {layer: X}, labels, turn_index=turn_idx, sae=sae,
                    sae_factory=sae_factory, groups=groups,
                )
                # Retain the LR per-layer results too. Computed with the SAME
                # sae_factory as the mass-mean call, so the two AUC curves are
                # the same (leakage-free, per-fold) estimator and are directly
                # comparable on a shared axis. Used only for the supplementary
                # AUC-by-depth diagnostic; selection (best_layer/best_turn) is
                # unchanged and still driven by mass-mean AUC below.
                lr_results += self.lr_probe.evaluate(
                    {layer: X}, labels, turn_index=turn_idx, sae=sae,
                    sae_factory=sae_factory, groups=groups,
                )
            results_by_turn[turn_idx]          = mm_results
            self._lr_results_by_turn[turn_idx] = lr_results
            self._log_turn_results(mm_results, turn_idx)

            # ── Raw-space probes (sae=None → evaluate in residual stream) ─────
            # Parallel run on the same mean-pooled activations, no SAE encoding.
            # CV uses _cv_auc (standard stratified) rather than _cv_auc_sae_per_fold
            # because there is no per-fold SAE to fit. Results are stored for the
            # direction cosine similarity diagnostic computed after best-cell selection;
            # they do not affect selection (which is driven by SAE-space AUC).
            raw_mm_results: list = []
            raw_lr_results: list = []
            for layer, X in mean_acts.items():
                raw_mm_results += self.raw_probe.evaluate(
                    {layer: X}, labels, turn_index=turn_idx, groups=groups
                )
                raw_lr_results += self.raw_lr_probe.evaluate(
                    {layer: X}, labels, turn_index=turn_idx, groups=groups
                )
            self._raw_results_by_turn[turn_idx]    = raw_mm_results
            self._raw_lr_results_by_turn[turn_idx] = raw_lr_results

        # Store final-token activations for bias comparison reporting
        self._bias_diagnostics["final_token_acts_turn_last"] = final_acts

        # ── Best layer / turn (by mass-mean AUC) ──────────────────────────────
        best_turn  = max(results_by_turn, key=lambda t: max(r.auc for r in results_by_turn[t]))
        best_layer = max(results_by_turn[best_turn], key=lambda r: r.auc).layer
        logger.info("Best separation: layer=%s turn=%d", best_layer, best_turn)

        # The turn loop leaves each probe's stored direction at the LAST turn.
        # Re-fit the direction at best_turn so the causal stage below patches the
        # same turn that won probe selection (no sae_factory: we only need the
        # deployable direction here, not another CV pass).
        best_mean_acts, best_final_acts, best_labels = self.collector.collect_dual(
            payloads, self.environment, turn_index=best_turn
        )
        sae_best = self._saes[best_layer]
        self.probe.evaluate(
            {best_layer: best_mean_acts[best_layer]}, best_labels,
            turn_index=best_turn, sae=sae_best, groups=groups,
        )
        self.lr_probe.evaluate(
            {best_layer: best_mean_acts[best_layer]}, best_labels,
            turn_index=best_turn, sae=sae_best, groups=groups,
        )
        # Re-fit raw-space probe directions at best_turn on the same activations
        # (no SAE). This ensures direction_for(best_layer) on the raw probes
        # corresponds to the same turn as the SAE-space direction, so the cosine
        # similarity comparison is not contaminated by turn mismatch.
        self.raw_probe.evaluate(
            {best_layer: best_mean_acts[best_layer]}, best_labels,
            turn_index=best_turn, groups=groups,
        )
        self.raw_lr_probe.evaluate(
            {best_layer: best_mean_acts[best_layer]}, best_labels,
            turn_index=best_turn, groups=groups,
        )

        # ── Causal comparison — selects the more causal probe direction ───────
        # Aligned to best_turn (tokens and legitimate activations both at best_turn).
        legit_acts, _ = self.collector.collect(
            [p for p in payloads if p.label == 0], self.environment, turn_index=best_turn
        )
        # compare_probes expects per-layer MEAN legitimate activations, not the
        # full stack of per-sample activations.
        legit_mean_acts = {layer: X.mean(dim=0) for layer, X in legit_acts.items()}
        causal_comparison = compare_probes(
            mass_mean_probe=self.probe,
            lr_probe=self.lr_probe,
            patcher=self.patching,
            adv_payloads=[p for p in payloads if p.label == 1],
            legit_mean_acts=legit_mean_acts,
            env=self.environment,
            best_layer=best_layer,
            turn_index=best_turn,
        )

        # ── Selection-adjusted headline AUC (winner's curse control) ──────────
        # The best cell's CV AUC is optimistically biased by argmax over the
        # layer x turn grid. Re-estimate it leakage-free on a FRESH draw so the
        # reported headline is not the selection-inflated number.
        selected_cell = self._reestimate_selected_cell(best_layer, best_turn, n)

        # ── Pooling robustness band (secondary; BC3 already dual-collects) ────
        # Report the NON-primary pooling's AUC at the selected cell alongside the
        # primary. Does not change the pre-registered per-class primary pooling
        # (§4.12); a large gap is reported, not acted on.
        pooling_band = self._pooling_band(
            best_layer, best_turn, best_mean_acts, best_final_acts, best_labels,
            groups=groups,
        )

        best_result = max(results_by_turn[best_turn], key=lambda r: r.auc)
        raw_space   = self._raw_space_comparison(best_layer, best_turn,
                                                  results_by_turn,
                                                  best_mean_acts[best_layer],
                                                  best_labels, groups=groups)
        return {
            "run_name":          self.cfg.run_name,
            "mode":              self.mode.value,
            "n_pairs":           n,
            "best_layer":        best_layer,
            "best_turn":         best_turn,
            "best_cell_auc":     {
                "auc":            best_result.auc,
                "auc_cv_std":     best_result.auc_cv_std,
                "auc_95ci":       _normal_ci(best_result.auc, best_result.auc_cv_std),
                "note":           "selection-inflated (argmax over layer x turn); see selection_adjusted",
            },
            "selection_adjusted": selected_cell,
            "pooling_band":      pooling_band,
            "accumulation":      self._accumulation_summary(results_by_turn),
            "layer_heatmap":     self._heatmap_summary(results_by_turn),
            "layer_heatmap_lr":  self._heatmap_summary(self._lr_results_by_turn),
            "layer_heatmap_raw": self._heatmap_summary(self._raw_results_by_turn),
            "patching":          self._run_mean_patching(payloads, best_layer, legit_acts, best_turn),
            "causal_comparison": causal_comparison,
            "raw_space":         raw_space,
            "run_manifest":      self._run_manifest(n),
            "timestamp":         datetime.now(timezone.utc).isoformat(),
        }

    def _raw_space_comparison(
        self,
        best_layer: str,
        best_turn: int,
        results_by_turn: dict,
        X_best: torch.Tensor,
        labels: torch.Tensor,
        groups=None,
    ) -> dict:
        """
        Direction cosine similarity between the SAE-space and raw-space probe
        directions at the selected (best_layer, best_turn) cell.

        Interpretation
        --------------
        The SAE-space mass-mean direction is fit on SAE-encoded activations and
        then projected back into residual space via the decoder
        (raw_direction_for = W_dec @ feature_dir, unit-normalised).
        The raw-space direction is fit directly on residual stream activations
        (no SAE encoding, no decoder projection).

        If these two raw-residual directions are closely aligned (cosine
        similarity near 1.0), the SAE is faithfully decomposing a feature that
        exists in the residual stream as a coherent linear direction — the SAE
        is recovering a real structure, not constructing an artifact of the
        dictionary learning objective.

        If they diverge (cosine similarity near 0 or negative), the SAE
        feature direction does not correspond to the principal separating
        direction in raw space. The probe may still be predictive (AUC can be
        high in either space) but the mechanistic claim — that the SAE is
        revealing a genuine residual-stream intent direction — is unsupported.

        Both the mass-mean and logistic regression comparisons are reported
        because the two probe types can disagree on the raw-space direction
        (LR is data-dependent; mass-mean is deterministic).

        This is a diagnostic output only. It does not affect selection, patching,
        or any pre-registered threshold. A low cosine similarity should be
        reported as a limitation and considered in the amendment protocol.

        Raw-space AUC at the selected cell is also reported alongside the
        SAE-space AUC so the two can be compared directly. A large gap
        (raw AUC << SAE AUC) suggests the SAE is amplifying a weak signal,
        which is worth noting; a gap in the other direction (raw AUC >> SAE AUC)
        suggests the SAE encoding is compressing useful variance.
        """
        # ── Direction cosine similarities ─────────────────────────────────────
        # SAE-space direction projected to raw residual space via decoder.
        sae_raw_dir_mm = self.probe.raw_direction_for(best_layer)       # [d_model]
        sae_raw_dir_lr = self.lr_probe.raw_direction_for(best_layer)    # [d_model]

        # Raw-space directions (already in residual space; direction_for == raw_direction_for).
        raw_dir_mm = self.raw_probe.direction_for(best_layer)           # [d_model]
        raw_dir_lr = self.raw_lr_probe.direction_for(best_layer)        # [d_model]

        cos_mm = float((sae_raw_dir_mm @ raw_dir_mm).item())
        cos_lr = float((sae_raw_dir_lr @ raw_dir_lr).item())

        logger.info(
            "Direction cosine similarity at %s (turn %d): "
            "mass-mean=%.4f, logistic-regression=%.4f",
            best_layer, best_turn, cos_mm, cos_lr,
        )

        # ── Raw-space AUC at the selected cell ────────────────────────────────
        # Retrieve from the stored per-turn raw results (already computed in
        # the turn loop). Fall back to re-evaluating if the turn wasn't stored
        # (e.g. best_turn differs from max_turns - 1 after re-fit).
        raw_turn_results = self._raw_results_by_turn.get(best_turn, [])
        raw_result_mm = next(
            (r for r in raw_turn_results if r.layer == best_layer), None
        )
        if raw_result_mm is None:
            # Re-evaluate at best_turn if not cached (should be rare).
            raw_result_mm = self.raw_probe.evaluate(
                {best_layer: X_best}, labels, turn_index=best_turn, groups=groups
            )[0]

        # SAE-space AUC at the selected cell for direct comparison.
        sae_turn_results = results_by_turn.get(best_turn, [])
        sae_result_mm = next(
            (r for r in sae_turn_results if r.layer == best_layer), None
        )
        sae_auc = sae_result_mm.auc if sae_result_mm is not None else None

        auc_gap = (
            round(sae_auc - raw_result_mm.auc, 4)
            if sae_auc is not None else None
        )

        return {
            "layer":                   best_layer,
            "turn":                    best_turn,
            "direction_cosine_sim": {
                "mass_mean":           round(cos_mm, 4),
                "logistic_regression": round(cos_lr, 4),
                "note": (
                    "Cosine similarity between SAE-decoder-projected direction and "
                    "raw-residual direction. Near 1.0 → SAE recovers a real "
                    "residual-stream feature. Near 0 → SAE direction is an artifact "
                    "of dictionary learning, not a coherent residual direction."
                ),
            },
            "auc_comparison": {
                "sae_space_auc":  sae_auc,
                "raw_space_auc":  raw_result_mm.auc,
                "raw_auc_cv_std": raw_result_mm.auc_cv_std,
                "gap_sae_minus_raw": auc_gap,
                "note": (
                    "SAE-space vs raw-residual-space CV AUC at the selected cell. "
                    "gap > 0 → SAE amplifies signal; gap < 0 → SAE compresses it."
                ),
            },
            "layer_heatmap_raw_mm":    self._heatmap_summary(self._raw_results_by_turn),
            "layer_heatmap_raw_lr":    self._heatmap_summary(self._raw_lr_results_by_turn),
        }

    def _pooling_band(self, best_layer, best_turn, mean_acts, final_acts, labels,
                      groups=None) -> dict:
        """SECONDARY: probe AUC under both poolings at the selected cell, so the
        result's dependence on the pooling choice is visible. The pre-registered
        primary pooling (per class, §4.12) is unchanged; this only reports the band."""
        out = {}
        sae = self._saes.get(best_layer)
        for name, acts in (("mean_pooled", mean_acts), ("final_token", final_acts)):
            if best_layer not in acts:
                continue
            res = self.probe.evaluate(
                {best_layer: acts[best_layer]}, labels, turn_index=best_turn, sae=sae,
                groups=groups,
            )[0]
            out[name] = {"auc": res.auc_in_sample}
        if "mean_pooled" in out and "final_token" in out:
            out["abs_gap"] = round(abs(out["mean_pooled"]["auc"] - out["final_token"]["auc"]), 4)
        out["note"] = "in-sample AUC per pooling; primary pooling is pre-registered per class (§4.12)"
        return out

    def _run_manifest(self, n_pairs: int) -> dict:
        """Reproducibility manifest (§8 infrastructure; non-confirmatory). Captures
        the seed, package/runtime versions, model revision, and config so a run is
        independently reproducible."""
        import platform
        manifest = {
            "seed":            getattr(self.cfg, "seed", None),
            "n_pairs":         n_pairs,
            "mode":            self.mode.value,
            "primary_model":   getattr(self.cfg, "model_name", "EleutherAI/pythia-1.4b"),
            "model_revision":  getattr(self.cfg, "model_revision", None),
            "sae_n_features":  self.sae_n_features,
            "sae_l1_coeff":    self.sae_l1_coeff,
            "sae_n_steps":     self.sae_n_steps,
            "cv_sae_n_steps":  self.cv_sae_n_steps,
            "python":          evidence_platform.python_version(),
            "torch":           getattr(torch, "__version__", None),
            "numpy":           getattr(np, "__version__", None),
            "cuda":            torch.cuda.is_available() if hasattr(torch, "cuda") else None,
            "dataset_provenance": getattr(self.environment, "dataset_provenance", None),
            "layer_scan":      getattr(self, "_layer_validation", None),
            "timestamp":       datetime.now(timezone.utc).isoformat(),
        }
        try:
            import transformer_lens  # noqa
            manifest["transformer_lens"] = getattr(transformer_lens, "__version__", "present")
        except Exception:
            manifest["transformer_lens"] = None
        return manifest

    def _reestimate_selected_cell(self, best_layer: str, best_turn: int, n: int) -> dict:
        """Leakage-free re-estimate of the selected (best_layer, best_turn) cell on
        a FRESH, independently drawn dataset — de-biases the argmax-selected
        headline AUC (winner's curse). Uses the per-fold SAE CV path.

        IMPORTANT: deliberately does NOT fall back to self._saes[best_layer].
        The deployable SAE was fit on the original N samples; reusing it as the
        encoder for a supposedly independent fresh draw would contaminate the
        re-estimate because the dictionary geometry has already absorbed the
        original distribution. Instead, a fresh sae_factory is passed so every
        CV fold on the fresh data trains its own SAE from scratch — no information
        from the original dataset enters the re-estimate through the encoder.
        """
        fresh = self.environment.generate_pairs(n)   # continues the RNG stream → disjoint draw
        mean_acts, _, labels = self.collector.collect_dual(fresh, self.environment, turn_index=best_turn)
        groups = [p.template_key for p in fresh]     # frame-level CV (see run())
        if best_layer not in mean_acts:
            return {"error": f"best_layer {best_layer} absent in re-estimate collection"}
        # Construct a fresh factory bound to this call's local state — no reference
        # to self._saes, so the original dataset cannot leak through the encoder.
        fresh_sae_factory = self._make_sae_factory()
        res = self.probe.evaluate(
            {best_layer: mean_acts[best_layer]}, labels,
            turn_index=best_turn, sae=None, sae_factory=fresh_sae_factory,
            groups=groups,
        )[0]
        logger.info("Selection-adjusted AUC (fresh draw, SAE-independent): %.4f ± %.4f",
                    res.auc, res.auc_cv_std)
        return {
            "layer":      best_layer,
            "turn":       best_turn,
            "auc":        res.auc,
            "auc_cv_std": res.auc_cv_std,
            "auc_95ci":   _normal_ci(res.auc, res.auc_cv_std),
            "n_pairs":    n,
            "note":       (
                "independent draw; SAE fit per fold from fresh activations only — "
                "no deployable SAE reuse. Unbiased re-estimate of the selected cell."
            ),
        }

    def _validate_scan_layers(self) -> dict:
        """Filter the collector's hook_names to layers that exist in this model.

        Layer indices >= model.cfg.n_layers are dropped — the model-variation
        guard for cases like the registered scan set [4,8,12,16,20,24] on
        Pythia-1.4B (24 layers, valid indices 0-23, so 24 is out of range).
        Logs a warning, updates self.collector.hook_names in place, and returns a
        record for the run manifest. Raises if no requested layer is valid.

        NOTE (pre-registration): dropping the out-of-range top layer changes the
        scan set actually run vs the registered one. For the CONFIRMATORY run,
        decide deliberately whether to (a) accept the in-range subset, or (b)
        amend the registered set to a depth-appropriate spacing for this model
        (e.g. ...,20,23 for a 24-layer model). This guard implements (a); the
        manifest records exactly which layers were scanned.
        """
        n_layers = getattr(self.model.cfg, "n_layers", None)
        requested = list(self.collector.hook_names)
        if n_layers is None:
            return {"adjusted": False, "reason": "n_layers unknown", "used": requested}
        kept, dropped = [], []
        for hn in requested:
            m = re.match(r"blocks\.(\d+)\.", hn)
            idx = int(m.group(1)) if m else None
            (kept if (idx is None or idx < n_layers) else dropped).append(hn)
        if not kept:
            raise ValueError(
                f"No requested scan layers valid for a {n_layers}-layer model: {requested}"
            )
        if dropped:
            logger.warning(
                "Scan layers out of range for %d-layer model — dropped %s; scanning %s",
                n_layers, dropped, kept,
            )
            self.collector.hook_names = kept
        return {"adjusted": bool(dropped), "n_layers": n_layers,
                "requested": requested, "used": kept, "dropped": dropped}

    def refine_layers(self, report: dict, span: int = 2, n_pairs: Optional[int] = None) -> dict:
        """
        Finer-grid layer refinement (PREREGISTRATION.md §3.3): evaluate the
        ±`span` layers around the best layer from the coarse scan, on the same
        dataset, and return the refined best layer.

        Pre-registered procedure: "Once the best layer per class is identified,
        a finer grid of ±2 layers is evaluated in the final run." This method is
        the push-button implementation for the GLOBAL best layer; for per-class
        refinement, call it on a class-specific experiment.

        New layers' SAEs are fitted with the experiment's locked hyperparameters
        and cached alongside the coarse-scan SAEs, so downstream consumers
        (compare_probes, bias controls, transfer) can use the refined layer
        transparently via experiment._saes.

        Args:
            report:  The report dict returned by run() (uses report['best_layer']).
            span:    Layers either side of the best layer to evaluate (default ±2).
            n_pairs: Dataset size; defaults to the run-mode value (same as run()).

        Returns:
            Dict with refined_best_layer, refined_results (per layer: cv AUC),
            coarse_best_layer, and whether refinement changed the selection.
        """
        import re
        m = re.match(r"blocks\.(\d+)\.(.+)", report["best_layer"])
        if not m:
            raise ValueError(f"Cannot parse layer name: {report['best_layer']}")
        idx, suffix = int(m.group(1)), m.group(2)
        n_layers = self.model.cfg.n_layers
        fine_idxs = [i for i in range(idx - span, idx + span + 1)
                     if 0 <= i < n_layers and i != idx]
        fine_hooks = [f"blocks.{i}.{suffix}" for i in fine_idxs]
        all_hooks  = fine_hooks + [report["best_layer"]]   # include coarse best for comparison
        logger.info("Layer refinement (±%d): evaluating %s", span, all_hooks)

        n = n_pairs if n_pairs is not None else self._n_pairs
        self.environment.reset_rng()                       # same dataset as the main run
        payloads = self.environment.generate_pairs(n)

        fine_collector = ActivationCollector(self.model, all_hooks,
                                             batch_size=self.collector.batch_size)
        mean_acts, _, labels = fine_collector.collect_dual(payloads, self.environment, turn_index=-1)
        groups = [p.template_key for p in payloads]   # frame-level CV (see run())

        refined_results = {}
        for layer, X in mean_acts.items():
            sae = self._get_or_fit_sae(layer, X, labels=labels)
            res = self.probe.evaluate({layer: X}, labels, turn_index=-1, sae=sae, groups=groups)[0]
            self.lr_probe.evaluate({layer: X}, labels, turn_index=-1, sae=sae, groups=groups)
            refined_results[layer] = {"auc": res.auc, "auc_in_sample": res.auc_in_sample}
            logger.info("  %s: cv AUC=%.4f", layer, res.auc)

        refined_best = max(refined_results, key=lambda l: refined_results[l]["auc"])
        changed = refined_best != report["best_layer"]
        logger.info("Refined best layer: %s (%s coarse best %s)",
                    refined_best, "replaces" if changed else "confirms",
                    report["best_layer"])
        return {
            "coarse_best_layer":  report["best_layer"],
            "refined_best_layer": refined_best,
            "selection_changed":  changed,
            "span":               span,
            "refined_results":    refined_results,
        }

    def _run_mean_patching(
        self,
        payloads: list[RoutingPayload],
        best_layer: str,
        legit_acts: dict[str, torch.Tensor],
        best_turn: int = -1,
        max_patch: Optional[int] = None,
    ) -> dict:
        """
        Baseline full-mean substitution patch. Replaces the adversarial activation
        entirely with the legitimate class mean and measures routing change rate.
        Distinct from directional patching used in compare_probes().

        The adversarial sample scales with the run rather than a fixed 20: by
        default the whole adversarial set is patched (cap via max_patch), and a
        Wilson 95% CI is reported so the causal rate carries its uncertainty.
        Tokens are taken at best_turn to match the selected cell.

        Payloads whose turn count is <= best_turn are excluded (Option A filter):
        patching a single-turn payload at best_turn=2 would mix two different
        interventions under one causal rate, conflating the turn-2 estimand with
        a turn-0 measurement. Skipped counts are surfaced in the returned dict
        keyed by attack_class so the scope is explicit in the run report.
        """
        legit_resid_mean = legit_acts[best_layer].mean(dim=0)
        adv = [p for p in payloads if p.label == 1]
        if max_patch is not None:
            adv = adv[:max_patch]

        # ── Filter to payloads structurally compatible with best_turn ─────────
        evaluated       = []
        skipped_by_class: dict[str, int] = {}
        for p in adv:
            turns = self.environment.tokenize(p)
            if best_turn >= len(turns):
                cls = getattr(p, "attack_class", "unknown")
                skipped_by_class[cls] = skipped_by_class.get(cls, 0) + 1
            else:
                evaluated.append(p)

        if skipped_by_class:
            logger.info(
                "Mean patching: skipped %d payloads incompatible with best_turn=%d: %s",
                sum(skipped_by_class.values()), best_turn, skipped_by_class,
            )

        changed = 0
        for payload in evaluated:
            tokens             = self.environment.tokenize(payload)[best_turn]
            unpatched, patched = self.patching.patch_and_run(
                tokens, legit_resid_mean, best_layer
            )
            if self.patching.routing_changed(unpatched, patched):
                changed += 1

        total       = len(evaluated)
        causal_rate = changed / total if total else 0.0
        lo, hi      = _wilson_ci(changed, total)
        logger.info("Mean patching: %d/%d routing changes (%.1f%%, 95%% CI %.1f-%.1f%%)",
                    changed, total, causal_rate * 100, lo * 100, hi * 100)
        return {
            "changed":          changed,
            "total":            total,
            "causal_rate":      round(causal_rate, 4),
            "causal_rate_95ci": [round(lo, 4), round(hi, 4)],
            "probe_layer":      best_layer,
            "turn":             best_turn,
            "skipped_by_class": skipped_by_class,
        }

    def _log_turn_results(self, results: list[ProbeResult], turn_idx: int) -> None:
        for r in results:
            try:
                wandb.log({"turn": turn_idx, "layer": r.layer,
                           "probe_acc": r.accuracy, "probe_auc": r.auc})
            except Exception:
                pass

    def _accumulation_summary(self, results_by_turn: dict) -> list[dict]:
        """Mean and max AUC across layers per turn — the accumulation curve."""
        return [
            {"turn":     turn_idx,
             "mean_auc": round(sum(r.auc for r in results) / len(results), 4),
             "max_auc":  round(max(r.auc for r in results), 4)}
            for turn_idx, results in sorted(results_by_turn.items())
        ]

    def _heatmap_summary(self, results_by_turn: dict) -> list[dict]:
        """AUC per layer per turn — the heatmap data."""
        return [
            {"turn": turn_idx, "layer": r.layer, "auc": r.auc, "accuracy": r.accuracy}
            for turn_idx, results in sorted(results_by_turn.items())
            for r in results
        ]

    def export(self, report: dict, out_path: Optional[Path] = None) -> Path:
        out_path = out_path or self.cfg.base_dir / "maritime_probe_results.jsonl"
        out_path.write_text(json.dumps(report, indent=2))
        logger.info("✓ Experiment report exported: %s", out_path)
        return out_path
