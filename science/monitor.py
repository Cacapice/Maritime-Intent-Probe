"""
monitor.py
AGPL 3. Copyright © 2026 Katherine J. Ombrellaro.
==========
DualLayerAuditMonitor: runtime deployment of the trained probe inside live
hook infrastructure.

"Dual-layer" = gateway filter (upstream, e.g. LiteLLM) + mechanistic probe
(internal, firing at the residual stream layer identified by ProbeExperiment).

The probe scores each turn by projecting SAE features onto the mass-mean
direction learned during the experimental phase. Non-destructive: logits are
returned unchanged, so the monitor can be inserted into any live pipeline
without altering model behaviour.

Deployment sequence:
  1. Run ProbeExperiment.run() to identify best_layer and fit SAE + probe.
  2. Save state with vault.save_probe_state() for persistence.
  3. Construct DualLayerAuditMonitor with the fitted SAE and probe direction.
  4. Call run_audit(payload, turn_id) for each incoming turn.
  5. Call summary() at end of session for aggregate statistics.
"""

import json
import logging
from datetime import datetime, timezone

import torch
import wandb

from science.config import VaultConfig
from science.dtypes import to_analysis, match
from science.sae import SparseAutoencoder

logger = logging.getLogger(__name__)


class DualLayerAuditMonitor:
    """
    Runtime deployment of the trained probe inside live hook infrastructure.

    Args:
        model:            Loaded HookedTransformer.
        cfg:              VaultConfig (for ledger path).
        sae:              Fitted SparseAutoencoder for the best layer.
        probe_direction:  Unit probe direction from MassMeanProbe.direction_for().
        best_layer:       Hook point where probe fires most reliably.
        intent_threshold: Score above which a turn is flagged (default 0.0).
        log_every_n:      W&B logging frequency (throughput control).
    """

    def __init__(
        self,
        model,
        cfg: VaultConfig,
        sae: SparseAutoencoder,
        probe_direction: torch.Tensor,
        best_layer: str,
        intent_threshold: float = 0.0,
        log_every_n: int = 1,
    ):
        self.model            = model
        self.cfg              = cfg
        self.sae              = sae
        self.probe_dir        = probe_direction
        self.best_layer       = best_layer
        self.intent_threshold = intent_threshold
        self.log_every_n      = log_every_n
        self._turn_scores: list[dict] = []      # scores for the CURRENT turn only
        self._session_scores: list[dict] = []   # accumulated across all turns

    def _intent_hook(self, activation: torch.Tensor, hook) -> torch.Tensor:
        """Mean-pool → SAE encode → dot with probe direction → flag if over threshold.
        Non-destructive: returns activation unchanged.
        """
        try:
            mean_act     = to_analysis(activation.mean(dim=1)).detach().cpu()
            features     = self.sae.encode(mean_act)
            intent_score = (features @ match(self.probe_dir, features)).mean().item()
        except Exception as e:
            logger.warning("Probe inference failed at %s: %s", hook.name, e)
            return activation

        entry = {"hook": hook.name, "intent_score": round(intent_score, 4)}
        self._turn_scores.append(entry)
        self._session_scores.append(entry)

        if intent_score > self.intent_threshold:
            logger.warning("⚠ Harmful intent detected | layer: %s | score: %.4f",
                           hook.name, intent_score)
        return activation

    def run_audit(self, payload: torch.Tensor, turn_id: int) -> torch.Tensor:
        """
        Audited forward pass at the best layer only.
        Gateway proxy check (LiteLLM) occurs upstream before this call.
        Returns unmodified logits.
        """
        self._turn_scores = []

        with self.model.hooks(fwd_hooks=[(self.best_layer, self._intent_hook)]):
            logits = self.model(payload)

        if turn_id % self.log_every_n == 0:
            self._log(turn_id)

        return logits

    def _log(self, turn_id: int) -> None:
        for entry in self._turn_scores:
            try:
                wandb.log({
                    "turn_id":      turn_id,
                    "hook":         entry["hook"],
                    "intent_score": entry["intent_score"],
                    "flagged":      entry["intent_score"] > self.intent_threshold,
                })
            except Exception:
                pass

        with open(self.cfg.ledger_path, "a") as f:
            for entry in self._turn_scores:
                f.write(json.dumps({
                    "type": "runtime_audit", "turn_id": turn_id,
                    **entry,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }) + "\n")

    def summary(self) -> dict:
        """Session-level intent score summary across ALL audited turns."""
        if not self._session_scores:
            logger.warning("summary() called with no recorded turns.")
            return {}
        scores  = [e["intent_score"] for e in self._session_scores]
        flagged = [s for s in scores if s > self.intent_threshold]
        return {
            "total_turns":   len(scores),
            "flagged_turns": len(flagged),
            "flag_rate":     round(len(flagged) / len(scores), 4),
            "mean_score":    round(sum(scores) / len(scores), 4),
            "max_score":     round(max(scores), 4),
        }
