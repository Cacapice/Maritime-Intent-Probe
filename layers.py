"""
layers.py — single source of truth for the residual-stream scan grid.
AGPL 3. Copyright © 2026 Katherine J. Ombrellaro.
=====================================================================
The probe / BC1 null gate / surface-geometry battery all scan a fixed set of
residual-stream layers. On the PRIMARY model (Pythia-1.4B, 24 layers) that set
is the pre-registered grid:

    pre-registration layers [4, 8, 12, 16, 20, 24]  (1-indexed)
      = 0-indexed TransformerLens blocks [3, 7, 11, 15, 19, 23]   (PRIMARY_SCAN)

The bug this module fixes
-------------------------
Those six indices were being reused *as raw absolute indices* on a
different-depth model. On Pythia-6.9B (32 layers) block 23 is only depth
fraction 23/32 ≈ 0.72, so applying [3,7,11,15,19,23] there silently drops the
deepest ~28% of the network — precisely the region where the 1.4B surface
signal concentrates (1.4B block 23 ≈ depth 0.96, its *final* layer). The BC1
null gate run on 6.9B at the raw grid therefore never sampled 6.9B's near-final
layers, so its worst-AUC could not see a depth-concentrated leak even if one
existed. `experiment._validate_scan_layers()` only *drops* out-of-range layers
(the smaller-model case); it never *adds* the deeper layers a larger model
affords.

What this module does
---------------------
`scan_layers_for(n_layers)` maps the primary grid onto a target model BY DEPTH
FRACTION, so the six scan points span the same depth fractions on every model:

    primary block i  →  round( i * n_layers / PRIMARY_N_LAYERS )   (clamped)

  • n_layers == 24  → identity → [3, 7, 11, 15, 19, 23]   (primary unchanged)
  • n_layers == 32  →            [4, 9, 15, 20, 25, 31]   (incl. the final block)

The depth-fraction convention (block_index / n_layers) is the SAME one used by
`surface_geometry.compare_across_models` and the notebook's per-depth plots, so
the collection grid and the cross-model overlay now agree point-for-point
instead of the overlay silently comparing a 0.09–0.72 trajectory (6.9B) against
a 0.12–0.96 one (1.4B). It is also the same proportional rule already used by
`scale_probe.py`'s single-layer mapper, `round(layer_1b * 32 / 24)`.

Locked-threshold note
---------------------
This does NOT alter a locked inferential threshold. On the primary model the
grid is byte-identical to the pre-registered set; for a scale replication,
depth-proportional layer mapping is the pre-registration's own rule for the
6.9B comparison (PREREGISTRATION.md §3.7 / §4.7), here applied consistently to
the geometry/BC1 path rather than only to the targeted scale probe. Re-running
the 6.9B BC1 gate / geometry on the depth-matched grid WILL change the recorded
6.9B numbers — that re-run is the actual test of whether the previous near-chance
6.9B worst-AUC was a deep-layer sampling gap or a genuine scale difference.
"""

from __future__ import annotations

# Pre-registered primary scan grid, 0-indexed (= 1-indexed [4,8,12,16,20,24]).
PRIMARY_SCAN: tuple[int, ...] = (3, 7, 11, 15, 19, 23)
PRIMARY_N_LAYERS: int = 24   # Pythia-1.4B, the primary model these indices lock to.


def scan_layers_for(
    n_layers: int,
    *,
    primary: tuple[int, ...] = PRIMARY_SCAN,
    primary_n_layers: int = PRIMARY_N_LAYERS,
    dense: bool = False,
) -> tuple[int, ...]:
    """Depth-matched 0-indexed scan layers for a model with `n_layers` blocks.

    Identity on the primary model (returns `primary` unchanged when
    n_layers == primary_n_layers), depth-fraction-mapped otherwise. The mapping
    preserves block/n_layers depth fraction and rounds half-up, then clamps to
    [0, n_layers-1] and de-duplicates preserving order — so the deepest primary
    layer always maps to the target model's near-final block (e.g. 23 → 31 on a
    32-layer model) and the deep region is no longer dropped.

    Args:
        n_layers:         Total blocks in the target model (e.g. model.cfg.n_layers).
        primary:          The locked primary grid to map (default PRIMARY_SCAN).
        primary_n_layers: Depth the primary grid is defined against (default 24).
        dense:            If True, return EVERY block index (0..n_layers-1) — the
                          "all layers" exploratory mode for a finer depth-fraction
                          trajectory. The six-point grid is the default.

    Returns:
        Tuple of 0-indexed block indices, ascending.
    """
    if not isinstance(n_layers, int) or n_layers <= 0:
        raise ValueError(f"n_layers must be a positive int, got {n_layers!r}")

    if dense:
        return tuple(range(n_layers))

    if n_layers == primary_n_layers:
        return tuple(primary)   # primary model — locked grid, unchanged.

    seen: set[int] = set()
    mapped: list[int] = []
    for idx in primary:
        # depth fraction idx/primary_n_layers preserved; round half-up, then clamp.
        j = int(idx * n_layers / primary_n_layers + 0.5)
        j = max(0, min(n_layers - 1, j))
        if j not in seen:
            seen.add(j)
            mapped.append(j)
    return tuple(mapped)


def hook_names_for(
    n_layers: int,
    *,
    hook: str = "hook_resid_post",
    **kwargs,
) -> list[str]:
    """Convenience: the depth-matched scan layers as TransformerLens hook names,
    e.g. ['blocks.4.hook_resid_post', ...]. Extra kwargs forward to
    `scan_layers_for` (e.g. dense=True)."""
    return [f"blocks.{i}.{hook}" for i in scan_layers_for(n_layers, **kwargs)]
