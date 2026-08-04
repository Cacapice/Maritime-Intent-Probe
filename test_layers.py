"""Regression tests for the depth-matched scan grid."""
from science.layers import PRIMARY_N_LAYERS, PRIMARY_SCAN, hook_names_for, scan_layers_for


def test_primary_grid_is_locked():
    assert scan_layers_for(PRIMARY_N_LAYERS) == PRIMARY_SCAN
    assert hook_names_for(24) == [f"blocks.{i}.hook_resid_post" for i in PRIMARY_SCAN]


def test_69b_includes_deep_layer():
    assert scan_layers_for(32) == (4, 9, 15, 20, 25, 31)


def test_depth_fraction_alignment():
    for primary, mapped in zip(PRIMARY_SCAN, scan_layers_for(32)):
        assert abs(primary / 24 - mapped / 32) <= 1 / 32 + 1e-9


def test_grids_are_well_formed():
    for n_layers in (24, 32, 36, 48):
        grid = scan_layers_for(n_layers)
        assert tuple(sorted(set(grid))) == grid
        assert grid
        assert all(0 <= i < n_layers for i in grid)


def test_dense_mode_returns_every_block():
    assert scan_layers_for(32, dense=True) == tuple(range(32))
