"""
dtypes.py — single dtype boundary for the analysis pipeline.
AGPL 3. Copyright © 2026 Katherine J. Ombrellaro.
============================================================
The model forward pass runs in its native dtype: float32 at 1.4B, bfloat16 at
6.9B. Every *stored* analysis tensor — probe directions, SAE weights, class-mean
vectors — is float32. Mixing them at a seam (e.g. `act @ direction` inside an
intervention hook) raises:

    RuntimeError: expected scalar type BFloat16 but found Float

Rather than scatter ad-hoc casts across every hook and scorer (and fix it one
crash at a time), we standardize TWO boundaries and route all dtype crossings
through them:

  1. READ-OUT  (forward pass → analysis): upcast activations to ANALYSIS_DTYPE
     the moment they leave the model. Everything downstream — probe fit, SAE
     fit, mean-diff directions, scoring — then lives in one dtype. Applied once,
     in the collector's caching hook.

  2. WRITE-BACK (analysis → forward pass): inside an intervention hook, downcast
     the edited activation back to the stream's dtype just before returning it.
     Applied in each patching hook.

Why hold analysis in float32 even when the model is bf16: the probe/SAE math and
the causal-effect *magnitudes* are compared against the float32 1.4B run. If the
pipeline silently dropped to bf16 mid-stream, the cross-model comparison would be
contaminated by precision, not geometry. Keep the compute in fp32; only the
forward pass itself is bf16 at 6.9B.

Usage
-----
    from dtypes import to_analysis, to_stream, match

    # collector read-out:
    self.cache[hook.name] = to_analysis(act.detach())

    # patching write-back:
    def hook(act, hook):
        a   = to_analysis(act)
        d   = match(direction, a)
        out = a - (a @ d).unsqueeze(-1) * d + match(legit_proj, a)
        return to_stream(out, act)
"""

import torch

# The dtype all stored analysis tensors and all analysis math live in. Do not
# change this to follow the model dtype — that is the whole point.
ANALYSIS_DTYPE: torch.dtype = torch.float32


def to_analysis(t: torch.Tensor) -> torch.Tensor:
    """Boundary 1 — upcast an activation as it leaves the forward pass.
    No-op when already ANALYSIS_DTYPE, so it is safe at 1.4B (float32) too."""
    return t.to(ANALYSIS_DTYPE)


def to_stream(t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Boundary 2 — cast an edited tensor back to the stream's dtype AND device
    before returning it from a hook. `like` is the original (unedited) `act`,
    so the replacement matches whatever TransformerLens handed in."""
    return t.to(dtype=like.dtype, device=like.device)


def match(stored: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Place a stored analysis tensor (direction / class-mean / legit component)
    on the activation's device, held in ANALYSIS_DTYPE for fp32 analysis math.
    Use this for the right-hand operands inside a hook so the matmul is fp32 on
    one device regardless of the model's dtype."""
    return stored.to(device=like.device, dtype=ANALYSIS_DTYPE)
