# Figures — Phase 1 geometry diagnostics

These figures describe representation geometry and probe-label separability. They do not license semantic interpretation of adversarial intent.

---

## 1.4B_geometry_depth_profile.png

**Question.** How do label separability, multi-direction advantage, and raw centroid distance vary with depth in the 1.4B model?  
**Observation.** Held-out AUC remains near 0.76 and above the permutation-null band across depth; the multi-direction advantage remains positive but narrows; raw centroid distance increases with depth. The final-layer projection shows visible class clustering under both direction constructions.  
**Interpretation.** Surface-label geometry is persistent and multi-direction structure adds predictive information, while the rising raw distance may reflect representational scale rather than stronger semantic separation.  
**Inference status.** Phase 1 descriptive geometry only: probe validity and scale-sensitive representation structure are observed; construct identification, semantic intent interpretation, and deployable monitoring remain unsupported.

---

## 6.9B_geometry_depth_profile.png

**Question.** Does the 6.9B model reproduce the same depth profile of surface-label geometry?  
**Observation.** Held-out AUC remains near 0.76–0.77 above the permutation null; multi-direction advantage stays positive with a late-depth dip; raw centroid distance grows strongly toward the final layer. Final-layer projections again show class clustering.  
**Interpretation.** The larger model reproduces the qualitative surface-separability pattern, but raw-distance growth is not a construct-validity result and may be scale-driven.  
**Inference status.** Phase 1 cross-scale descriptive replication only. It does not complete the crossed 2×2 experiment, identify adversarial intent, or validate a deployable monitor.
