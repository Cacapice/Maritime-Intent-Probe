# Local figure audit — Maritime Intent Probe (Phase 1)

## Scope

This audit applies only to the two geometry-depth profile figures committed under `figures/`. Maritime Intent Probe remains Phase 1 construct-validity diagnostic research.

## Inventory

- `1.4B_geometry_depth_profile.png`
- `6.9B_geometry_depth_profile.png`

Each figure now has a complete caption in `docs/figures/FIGURES.md` and a `.figure.json` sidecar containing dimensions, SHA-256 hash, source status, research phase, lifecycle, and the Phase 1 inference boundary.

## Interpretation boundary

The figures support descriptive claims about representation geometry, held-out label separability, depth profiles, and scale-sensitive centroid distance. They do not identify adversarial intent, validate a deployable monitor, or complete the crossed 2×2 redesign. The artifact tests enforce this boundary in both caption text and metadata.

## Release status

Artifact-level tests reject missing captions, missing metadata, hash mismatches, and any metadata record that omits the Phase 1 semantic-interpretation block.
