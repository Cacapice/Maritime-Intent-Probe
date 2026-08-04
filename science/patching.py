"""
patching.py
AGPL 3. Copyright © 2026 Katherine J. Ombrellaro.
===========
CausalPatcher: activation patching to confirm that the probed representation
causally mediates routing outputs, not just correlates with them.

Two patching methods:

  patch_and_run()            — replaces the full activation with the legitimate
                               class mean. Used for the baseline causal check.

  directional_patch_and_run() — targeted directional patch: subtracts the
                               adversarial component along a probe direction
                               and replaces it with the legitimate mean
                               projection along that direction. Used in
                               compare_probes() to adjudicate which probe
                               direction is more causally upstream of routing
                               outputs — the empirical test of the Marks &
                               Tegmark hypothesis in this domain.

Reference: Meng et al. (2022) "Locating and Editing Factual Associations in GPT"
for the general activation patching methodology.
"""

import logging
import re

import torch
from transformer_lens import HookedTransformer

from science.dtypes import to_analysis, to_stream

logger = logging.getLogger(__name__)


class CausalPatcher:
    """
    Activation patching at a target residual stream layer.
    Routing change = causal confirmation, not just correlation.
    """

    def __init__(self, model: HookedTransformer):
        self.model = model

    def _next_resid_layer(self, layer: str) -> str:
        """
        Given a hook name like 'blocks.12.hook_resid_post', return the next
        block's resid_post hook ('blocks.13.hook_resid_post') if it exists,
        else return the input layer unchanged (e.g. when patching the last block).
        """
        m = re.match(r"blocks\.(\d+)\.(.+)", layer)
        if not m:
            return layer
        idx, suffix = int(m.group(1)), m.group(2)
        n_layers = getattr(self.model.cfg, "n_layers", None)
        if n_layers is None or idx + 1 >= n_layers:
            return layer
        return f"blocks.{idx + 1}.{suffix}"

    @torch.no_grad()
    def patch_and_run(
        self,
        adversarial_tokens: torch.Tensor,
        legitimate_mean_act: torch.Tensor,
        layer: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Full mean substitution patch. Replaces the entire activation at `layer`
        with the legitimate class mean. Used for the baseline causal check.

        Returns: (unpatched_logits, patched_logits)
        """
        unpatched = self.model(adversarial_tokens).cpu()
        adversarial_tokens = adversarial_tokens.to(next(self.model.parameters()).device)
        patch_vec = legitimate_mean_act.to(adversarial_tokens.device)

        def patch_hook(act, hook):
            act[:, :, :] = to_stream(patch_vec, act).unsqueeze(0).unsqueeze(0)
            return act

        with self.model.hooks(fwd_hooks=[(layer, patch_hook)]):
            patched = self.model(adversarial_tokens).cpu()

        return unpatched, patched

    @torch.no_grad()
    def directional_patch_and_run(
        self,
        adversarial_tokens: torch.Tensor,
        legitimate_mean_act: torch.Tensor,
        probe_direction: torch.Tensor,
        layer: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Targeted directional patch along a specific probe direction.

        For each token position across the full sequence, the component of the
        activation along `probe_direction` is replaced with the corresponding
        component of `legitimate_mean_act`. All other activation components are
        unchanged.

        Patch position: applied uniformly across ALL token positions at the
        target layer [batch, seq_len, d_model]. This is consistent with
        final-token collection being the primary analysis path — the patch
        operates on the full sequence so that distributed signal (e.g.
        fragmentation attacks) is also addressed. The causal rate reflects
        routing output change after full-sequence directional suppression.

        This isolates whether the *specific axis* identified by the probe
        is causally responsible for the routing output, rather than any
        arbitrary change to the full activation. Used in compare_probes()
        to compare causal rates between MassMeanProbe and LogisticRegressionProbe
        directions on equal footing.

        Args:
            adversarial_tokens:  Token tensor [batch, seq_len].
            legitimate_mean_act: Mean legitimate residual stream activation [d_model].
            probe_direction:     Unit direction vector [d_model or n_features].
                                 Must be in the same space as the residual stream
                                 at `layer` (i.e. raw activation space, not SAE space).
            layer:               Hook name to patch at.

        Returns: (unpatched_logits, patched_logits)
        """
        unpatched    = self.model(adversarial_tokens).cpu()
        adversarial_tokens = adversarial_tokens.to(next(self.model.parameters()).device)
        direction    = probe_direction.to(adversarial_tokens.device).float()
        legit_mean   = legitimate_mean_act.to(adversarial_tokens.device).float()
        legit_proj   = (legit_mean @ direction) * direction   # legitimate component (fp32)

        def directional_hook(act, hook):
            # act: [batch, seq_len, d_model]; bf16 at 6.9B, fp32 at 1.4B.
            # Project in fp32 (analysis dtype), then cast back to the stream
            # dtype on return. No-op at 1.4B; fixes the bf16/fp32 matmul at 6.9B.
            a        = to_analysis(act)
            adv_proj = (a @ direction).unsqueeze(-1) * direction    # adversarial component
            return to_stream(a - adv_proj + legit_proj, act)        # swap component

        with self.model.hooks(fwd_hooks=[(layer, directional_hook)]):
            patched = self.model(adversarial_tokens).cpu()

        return unpatched, patched

    @torch.no_grad()
    def directional_patch_final_token_only(
        self,
        adversarial_tokens: torch.Tensor,
        legitimate_mean_act: torch.Tensor,
        probe_direction: torch.Tensor,
        layer: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Directional patch applied to the FINAL token position only.

        Step 1 of the pre-registered BC10 patch-scope fallback sequence
        (PREREGISTRATION.md §6): when the full-sequence directional patch
        degrades legitimate routing above threshold, narrowing the patch to the
        decision-bearing final position can preserve legitimate behaviour while
        still suppressing adversarial intent. Same direction/projection maths as
        directional_patch_and_run, restricted to position -1.

        Returns: (unpatched_logits, patched_logits)
        """
        unpatched  = self.model(adversarial_tokens).cpu()
        adversarial_tokens = adversarial_tokens.to(next(self.model.parameters()).device)
        direction  = probe_direction.to(adversarial_tokens.device).float()
        legit_mean = legitimate_mean_act.to(adversarial_tokens.device).float()
        legit_proj = (legit_mean @ direction) * direction

        def final_token_hook(act, hook):
            # act: [batch, seq_len, d_model]; patch only the last position.
            last = to_analysis(act[:, -1, :])                     # [batch, d_model] fp32
            adv_proj = (last @ direction).unsqueeze(-1) * direction
            act[:, -1, :] = to_stream(last - adv_proj + legit_proj, act)
            return act

        with self.model.hooks(fwd_hooks=[(layer, final_token_hook)]):
            patched = self.model(adversarial_tokens).cpu()

        return unpatched, patched

    @torch.no_grad()
    def per_position_causal_rates(
        self,
        adversarial_payloads: list,
        legitimate_mean_act: torch.Tensor,
        probe_direction: torch.Tensor,
        layer: str,
        env,
        max_positions: int = 8,
    ) -> dict:
        """
        Identify adversarial-signal-bearing token positions (BC10 fallback step 2,
        PREREGISTRATION.md §6).

        For each of the final `max_positions` token positions (indexed from the
        end, since payload lengths vary), the directional patch is applied to
        that single position and the fraction of adversarial routing outputs
        that flip is recorded. Positions whose individual causal rate is at
        least half the best position's rate are designated signal-bearing.

        Returns:
            Dict with rates_by_position ({-1: rate, -2: rate, ...}) and
            signal_positions (list of negative indices, always including -1's
            cohort if any position flips at all).
        """
        direction  = probe_direction.to(next(self.model.parameters()).device).float()
        legit_mean = legitimate_mean_act.to(direction.device).float()
        legit_proj = (legit_mean @ direction) * direction

        counts = {-(k + 1): 0 for k in range(max_positions)}
        totals = {-(k + 1): 0 for k in range(max_positions)}

        for payload in adversarial_payloads:
            tokens = env.tokenize(payload)[-1].to(direction.device)
            seq_len = tokens.shape[1]
            unpatched = self.model(tokens).cpu()

            for pos in counts:
                if -pos > seq_len:
                    continue
                totals[pos] += 1

                def pos_hook(act, hook, p=pos):
                    sl = to_analysis(act[:, p, :])
                    adv_proj = (sl @ direction).unsqueeze(-1) * direction
                    act[:, p, :] = to_stream(sl - adv_proj + legit_proj, act)
                    return act

                with self.model.hooks(fwd_hooks=[(layer, pos_hook)]):
                    patched = self.model(tokens).cpu()
                if self.routing_changed(unpatched, patched):
                    counts[pos] += 1

        rates = {pos: (counts[pos] / totals[pos] if totals[pos] else 0.0)
                 for pos in counts}
        best = max(rates.values()) if rates else 0.0
        signal_positions = sorted(
            [pos for pos, r in rates.items() if best > 0 and r >= 0.5 * best]
        )
        return {
            "rates_by_position": {str(k): round(v, 4) for k, v in rates.items()},
            "signal_positions":  signal_positions,
            "max_position_rate": round(best, 4),
        }

    @torch.no_grad()
    def directional_patch_positions_and_run(
        self,
        tokens: torch.Tensor,
        legitimate_mean_act: torch.Tensor,
        probe_direction: torch.Tensor,
        layer: str,
        positions: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Directional patch restricted to the given token positions (negative
        indices from the sequence end). BC10 fallback step 2 uses the
        signal-bearing positions identified by per_position_causal_rates().

        Returns: (unpatched_logits, patched_logits)
        """
        tokens     = tokens.to(next(self.model.parameters()).device)
        unpatched  = self.model(tokens).cpu()
        direction  = probe_direction.to(tokens.device).float()
        legit_mean = legitimate_mean_act.to(direction.device).float()
        legit_proj = (legit_mean @ direction) * direction
        seq_len    = tokens.shape[1]
        valid      = [p for p in positions if -p <= seq_len]

        def multi_pos_hook(act, hook):
            for p in valid:
                sl = to_analysis(act[:, p, :])
                adv_proj = (sl @ direction).unsqueeze(-1) * direction
                act[:, p, :] = to_stream(sl - adv_proj + legit_proj, act)
            return act

        with self.model.hooks(fwd_hooks=[(layer, multi_pos_hook)]):
            patched = self.model(tokens).cpu()

        return unpatched, patched

    def routing_changed(
        self,
        unpatched_logits: torch.Tensor,
        patched_logits: torch.Tensor,
    ) -> bool:
        """
        True if the routing DECISION changes.

        The decision is read from the final sequence position only — the
        position whose next-token prediction is the model's actual routing
        output. Comparing argmax across every position (the previous behaviour)
        flagged a change whenever any token anywhere shifted, which almost any
        intervention causes, inflating causal rates toward 100% and destroying
        the discriminative power of the H3 comparison.

        Logits may be [batch, seq, vocab] or [seq, vocab]; the final position
        along the sequence axis is used.
        """
        if unpatched_logits.dim() == 3:
            u = unpatched_logits[:, -1, :]
            p = patched_logits[:, -1, :]
        elif unpatched_logits.dim() == 2:
            u = unpatched_logits[-1:, :]
            p = patched_logits[-1:, :]
        else:
            u = unpatched_logits
            p = patched_logits
        return bool(u.argmax(-1).ne(p.argmax(-1)).any().item())

    @torch.no_grad()
    def post_patch_alignment_rate(
        self,
        adversarial_payloads: list,
        legitimate_mean_act: torch.Tensor,
        probe_direction: torch.Tensor,
        layer: str,
        env,
        collector,
        alignment_threshold: float = 0.70,
        measure_layer: str | None = None,
        legit_measure_mean_act: torch.Tensor | None = None,
    ) -> dict:
        """
        Post-patch alignment rate (pre-registered gap closure).

        For adversarial payloads where the directional patch produces a routing
        flip, measures whether the post-patch residual stream activation is
        geometrically aligned with the legitimate class mean — cosine similarity
        >= alignment_threshold.

        Two correctness points relative to the original implementation:

          1. The activation must be captured WITH THE PATCH APPLIED. The patch
             hook and the capture hook run in the same forward pass; capturing in
             a separate, unpatched pass (the previous behaviour) measured the
             adversarial activation, not the post-patch one.

          2. Alignment is measured at `measure_layer` (default: the layer
             immediately downstream of the patch layer if resolvable, else the
             patch layer). Measuring at the patch layer itself is near-tautological
             because the directional patch directly sets that activation's
             component along probe_direction. A downstream layer tests whether the
             intervention actually propagated the representation toward legitimate
             space.

        Args:
            adversarial_payloads:   Adversarial payloads to evaluate.
            legitimate_mean_act:    Mean legitimate residual activation at the PATCH
                                    layer [d_model] (used to build the patch).
            probe_direction:        Unit probe direction in raw residual space [d_model].
            layer:                  Hook name to patch at.
            env:                    MaritimeEnvironment (for tokenization).
            collector:              ActivationCollector (unused; kept for signature stability).
            alignment_threshold:    Cosine similarity threshold (default 0.70).
            measure_layer:          Hook name at which to measure alignment. Defaults
                                    to the next resid_post block after `layer`.
            legit_measure_mean_act: Legitimate class mean at measure_layer [d_model].
                                    If None, falls back to legitimate_mean_act (only
                                    valid when measure_layer == layer).

        Returns:
            Dict with causal_rate, alignment_rate, aligned_of_flipped,
            alignment_threshold, measure_layer, and interpretation.
        """
        device   = next(self.model.parameters()).device
        m_layer  = measure_layer or self._next_resid_layer(layer)
        legit_at_measure = (
            legit_measure_mean_act if legit_measure_mean_act is not None
            else legitimate_mean_act
        ).to(device).float()

        direction  = probe_direction.to(device).float()
        legit_mean = legitimate_mean_act.to(device).float()
        legit_proj = (legit_mean @ direction) * direction

        flipped = 0
        aligned_of_flipped = 0

        for payload in adversarial_payloads:
            if isinstance(payload, str):
                raise TypeError(
                    "post_patch_alignment_rate() expects RoutingPayload objects, "
                    "not strings. Pass payload objects directly — "
                    "env.tokenize() is called internally."
                )
            tokens = env.tokenize(payload)[-1].to(next(self.model.parameters()).device)
            unpatched, patched = self.directional_patch_and_run(
                tokens, legitimate_mean_act, probe_direction, layer
            )
            if not self.routing_changed(unpatched, patched):
                continue
            flipped += 1

            # Re-run with BOTH the directional patch and a downstream capture hook.
            captured = {}

            def patch_hook(act, hook):
                a = to_analysis(act)
                adv_proj = (a @ direction).unsqueeze(-1) * direction
                return to_stream(a - adv_proj + legit_proj, act)

            def capture_hook(act, hook):
                captured["act"] = to_analysis(act[:, -1, :].mean(dim=0)).detach().cpu()
                return act

            hooks = [(layer, patch_hook)]
            if m_layer != layer:
                hooks.append((m_layer, capture_hook))
            else:
                # Measuring at the patch layer: capture the patched activation.
                def patch_and_capture(act, hook):
                    a = to_analysis(act)
                    out = a - (a @ direction).unsqueeze(-1) * direction + legit_proj
                    captured["act"] = out[:, -1, :].mean(dim=0).detach().cpu()
                    return to_stream(out, act)
                hooks = [(layer, patch_and_capture)]

            with self.model.hooks(hooks):
                _ = self.model(tokens)

            if "act" in captured:
                act_vec = captured["act"].to(device)
                cos_sim = torch.nn.functional.cosine_similarity(
                    act_vec.unsqueeze(0), legit_at_measure.unsqueeze(0)
                ).item()
                if cos_sim >= alignment_threshold:
                    aligned_of_flipped += 1

        total = len(adversarial_payloads)
        causal_rate = flipped / total if total > 0 else 0.0
        alignment_rate = aligned_of_flipped / flipped if flipped > 0 else 0.0

        interpretation = (
            "HIGH CAUSAL RATE + HIGH ALIGNMENT — targeted intervention claim "
            "supported: patch suppresses adversarial intent and recovers "
            "legitimate routing behaviour."
            if causal_rate >= 0.20 and alignment_rate >= 0.70 else
            "HIGH CAUSAL RATE + LOW ALIGNMENT — suppression without recovery. "
            "Output changed but did not land in legitimate activation space. "
            "Limits production intervention interpretation."
            if causal_rate >= 0.20 and alignment_rate < 0.70 else
            "LOW CAUSAL RATE — directional patch does not reliably flip "
            "adversarial routing outputs at this threshold."
        )

        logger.info(
            "Post-patch alignment: causal_rate=%.1f%% alignment_rate=%.1f%% "
            "flipped=%d aligned_of_flipped=%d threshold=%.2f",
            causal_rate * 100, alignment_rate * 100,
            flipped, aligned_of_flipped, alignment_threshold,
        )

        return {
            "causal_rate":          round(causal_rate, 4),
            "alignment_rate":       round(alignment_rate, 4),
            "flipped":              flipped,
            "aligned_of_flipped":   aligned_of_flipped,
            "total_adversarial":    total,
            "alignment_threshold":  alignment_threshold,
            "measure_layer":        m_layer,
            "interpretation":       interpretation,
        }

    @torch.no_grad()
    def legitimate_specificity_check(
        self,
        legitimate_payloads: list,
        legitimate_mean_act: torch.Tensor,
        probe_direction: torch.Tensor,
        layer: str,
        env,
        degradation_threshold: float = 0.10,
    ) -> dict:
        """
        Legitimate payload specificity check (pre-registered gap closure).

        Applies the directional patch to legitimate payloads and measures
        whether their routing outputs remain stable. The non-destructive
        intervention claim requires empirical demonstration: the patch must
        suppress adversarial routing without degrading legitimate routing.

        Pre-registered criterion: degradation rate (fraction of legitimate
        payloads whose routing output changes after patching) must be <=
        degradation_threshold (default 0.10). A rate above threshold indicates
        the patch is suppressing something broader than adversarial intent.

        Args:
            legitimate_payloads:  Legitimate routing payloads to evaluate.
            legitimate_mean_act:  Mean legitimate residual stream activation [d_model].
            probe_direction:      Unit probe direction [d_model].
            layer:                Hook name to patch at.
            env:                  MaritimeEnvironment (for tokenization).
            degradation_threshold: Maximum acceptable legitimate degradation rate.

        Returns:
            Dict with degradation_rate, changed, total, threshold, passed flag,
            and interpretation.
        """
        changed = 0
        total = len(legitimate_payloads)

        for payload in legitimate_payloads:
            if isinstance(payload, str):
                raise TypeError(
                    "legitimate_specificity_check() expects RoutingPayload objects, "
                    "not strings. Pass payload objects directly — "
                    "env.tokenize() is called internally."
                )
            tokens = env.tokenize(payload)[-1].to(next(self.model.parameters()).device)
            unpatched, patched = self.directional_patch_and_run(
                tokens, legitimate_mean_act, probe_direction, layer
            )
            if self.routing_changed(unpatched, patched):
                changed += 1

        degradation_rate = changed / total if total > 0 else 0.0
        passed = degradation_rate <= degradation_threshold

        interpretation = (
            f"PASSED — legitimate degradation rate {degradation_rate:.1%} "
            f"<= {degradation_threshold:.1%} threshold. Non-destructive "
            f"intervention claim is empirically supported."
            if passed else
            f"FAILED — legitimate degradation rate {degradation_rate:.1%} "
            f"> {degradation_threshold:.1%} threshold. Patch is suppressing "
            f"something broader than adversarial intent. Limits production "
            f"intervention interpretation; report as limitation."
        )

        logger.info(
            "Legitimate specificity check: degradation_rate=%.1f%% "
            "changed=%d total=%d threshold=%.2f passed=%s",
            degradation_rate * 100, changed, total, degradation_threshold, passed,
        )

        return {
            "degradation_rate":      round(degradation_rate, 4),
            "changed":               changed,
            "total_legitimate":      total,
            "degradation_threshold": degradation_threshold,
            "passed":                passed,
            "interpretation":        interpretation,
        }
