"""
collector.py
AGPL 3. Copyright © 2026 Katherine J. Ombrellaro.
============
ActivationCollector: runs payloads through the model and captures residual
stream activations at specified hook points.

Two collection strategies are supported (Bias Control 3 — mean-pooling
misalignment, Elhage et al. 2021 / Kulkarni et al. 2026):

  collect()       — mean-pooled across all token positions (original,
                    retained for backward compatibility)
  collect_dual()  — returns BOTH mean-pooled AND final-token activations
                    so bias comparison runs without a second forward pass

Both methods use batched tokenisation and batched forward passes.
Each payload's target turn is tokenised, padded to a common length within
the batch, and passed through the model in a single GPU call. Mean-pooling
excludes pad positions via an attention mask. This replaces the original
per-payload sequential forward pass loop and produces a 4–8× speedup on
the collection step.

Hook names are passed in at construction time, so layer subset scoping
(for Colab iteration speed) is controlled entirely at the call site.
"""

import logging
from collections import defaultdict

import torch
from transformer_lens import HookedTransformer

from dtypes import to_analysis
from environment import MaritimeEnvironment, RoutingPayload

logger = logging.getLogger(__name__)

PAD_TOKEN_ID = 0   # Pythia tokeniser uses 0 as pad; override if using a different model


class ActivationCollector:
    """
    Runs payloads through the model and collects residual stream activations
    per layer per turn using batched inference.

    collect() returns mean-pooled activations: dict[layer_name → tensor[n_payloads, d_model]]
    collect_dual() returns both mean-pooled and final-token activations for
    bias comparison (Bias Control 3).

    Batching is handled internally. The public interface is unchanged from the
    sequential version — callers pass a list of payloads and receive stacked
    activation tensors.

    Layer subset example (Colab-friendly):
        hook_names = [f"blocks.{i}.hook_resid_post" for i in [3, 7, 11, 15, 19, 23]]
        collector  = ActivationCollector(model, hook_names)
    """

    def __init__(
        self,
        model: HookedTransformer,
        hook_names: list[str],
        batch_size: int = 16,
        pad_token_id: int = PAD_TOKEN_ID,
    ):
        self.model        = model
        self.hook_names   = hook_names
        self.batch_size   = batch_size
        self.pad_token_id = pad_token_id

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _prepare_batch(
        self,
        payloads: list[RoutingPayload],
        env: MaritimeEnvironment,
        turn_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenise each payload's target turn and pad to a common length.

        Returns:
            token_batch: [n_payloads, max_seq_len] — padded token IDs
            attn_mask:   [n_payloads, max_seq_len] — 1 for real tokens, 0 for pad
        """
        token_lists = []
        for payload in payloads:
            turns = env.tokenize(payload)
            idx   = (min(turn_index, len(turns) - 1) if turn_index >= 0
                     else len(turns) - 1)
            # env.tokenize returns list of [1, seq_len] tensors
            token_lists.append(turns[idx].squeeze(0).tolist())

        max_len = max(len(t) for t in token_lists)
        padded, masks = [], []
        for toks in token_lists:
            pad_len = max_len - len(toks)
            padded.append(toks + [self.pad_token_id] * pad_len)
            masks.append([1] * len(toks) + [0] * pad_len)

        return (
            torch.tensor(padded,  dtype=torch.long),
            torch.tensor(masks,   dtype=torch.float),
        )

    @torch.no_grad()
    def _run_batched(
        self,
        payloads: list[RoutingPayload],
        env: MaritimeEnvironment,
        turn_index: int,
        dual: bool = False,
    ) -> tuple[dict, dict | None]:
        """
        Core batched collection loop. Processes payloads in chunks of
        self.batch_size. Returns (mean_acts, final_acts | None).

        mean_acts:  dict[layer → list[Tensor[d_model]]]
        final_acts: dict[layer → list[Tensor[d_model]]] if dual else None

        Mean-pooling excludes pad positions using the attention mask, so
        activations at pad positions do not dilute the mean representation.
        """
        device     = next(self.model.parameters()).device
        mean_acts:  dict[str, list[torch.Tensor]] = defaultdict(list)
        final_acts: dict[str, list[torch.Tensor]] = defaultdict(list) if dual else None

        n = len(payloads)
        for start in range(0, n, self.batch_size):
            batch_payloads = payloads[start: start + self.batch_size]
            tokens, mask   = self._prepare_batch(batch_payloads, env, turn_index)
            tokens         = tokens.to(device)
            mask           = mask.to(device)          # [B, T]

            captured: dict[str, torch.Tensor] = {}

            def make_hook(name):
                def hook(act, **kwargs):
                    # act: [B, T, d_model]. Upcast to the analysis dtype at the
                    # READ-OUT boundary so everything downstream — pooling, probe
                    # fit, SAE fit, the class-mean directions — is float32
                    # regardless of the model's compute dtype (bf16 at 6.9B).
                    # No-op at 1.4B (already float32). See dtypes.py.
                    captured[name] = to_analysis(act.detach()).cpu()
                    return act
                return hook

            with self.model.hooks(
                fwd_hooks=[(hn, make_hook(hn)) for hn in self.hook_names]
            ):
                self.model(tokens)

            # Pool and store per-layer
            mask_cpu = mask.cpu().unsqueeze(-1)   # [B, T, 1]
            for name in self.hook_names:
                act = captured[name]              # [B, T, d_model]

                # Masked mean-pool: sum over real token positions, divide by count
                masked_sum  = (act * mask_cpu).sum(dim=1)               # [B, d_model]
                token_count = mask_cpu.sum(dim=1).clamp(min=1)          # [B, 1]
                mean_pooled = masked_sum / token_count                   # [B, d_model]

                for i in range(len(batch_payloads)):
                    mean_acts[name].append(mean_pooled[i])

                if dual:
                    # Final real token position per sample in the batch
                    # = last position where mask == 1
                    real_lengths = mask_cpu.squeeze(-1).sum(dim=1).long() - 1  # [B]
                    for i in range(len(batch_payloads)):
                        final_acts[name].append(act[i, real_lengths[i], :])

            logger.debug(
                "Collected batch %d–%d / %d",
                start, start + len(batch_payloads) - 1, n,
            )

        return mean_acts, final_acts

    # ── Public API ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def collect(
        self,
        payloads: list[RoutingPayload],
        env: MaritimeEnvironment,
        turn_index: int = -1,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        Collect masked mean-pooled activations for all payloads at the
        specified turn using batched inference.

        Args:
            payloads:   List of RoutingPayload objects to process.
            env:        MaritimeEnvironment used for tokenization.
            turn_index: Which turn to collect from. -1 = last turn (default).

        Returns:
            Tuple of (mean-pooled activations dict, labels tensor).
        """
        mean_acts, _ = self._run_batched(payloads, env, turn_index, dual=False)
        labels       = torch.tensor([p.label for p in payloads])
        return (
            {name: torch.stack(vecs) for name, vecs in mean_acts.items()},
            labels,
        )

    @torch.no_grad()
    def collect_dual(
        self,
        payloads: list[RoutingPayload],
        env: MaritimeEnvironment,
        turn_index: int = -1,
    ) -> tuple[
        dict[str, torch.Tensor],   # mean-pooled (mask-corrected)
        dict[str, torch.Tensor],   # final real token
        torch.Tensor,              # labels
    ]:
        """
        Collect BOTH masked mean-pooled and final-token activations in a
        single batched forward pass per chunk (Bias Control 3).

        In decoder-only transformers the final token position is the only
        position that has attended over the entire prompt and therefore carries
        the model's full contextual encoding prior to generation (Elhage et al.
        2021). For contextual priming attacks, mean-pooling dilutes this signal;
        for fragmentation, mean-pooling may aggregate distributed signal.

        Mean-pooling uses the attention mask to exclude pad positions.
        Final-token is the last real (non-pad) position per sample.

        Returns:
            (mean_acts, final_acts, labels)
            Each acts dict: layer_name → tensor[n_payloads, d_model]
        """
        mean_acts, final_acts = self._run_batched(
            payloads, env, turn_index, dual=True
        )
        labels = torch.tensor([p.label for p in payloads])
        return (
            {name: torch.stack(vecs) for name, vecs in mean_acts.items()},
            {name: torch.stack(vecs) for name, vecs in final_acts.items()},
            labels,
        )
