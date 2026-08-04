# Figure Standard: Graphics as Arguments

This repository treats every graphic as part of the inferential record, not as decoration. The standard follows the attached *Developing Graphics* guidance: choose the form from the analytical purpose, make the central comparison legible, remove visual material that does not support the discussion, and verify curves, scales, labels, and captions against the underlying data.

## Required figure contract

Every publication figure must declare:

1. **Question** — the single question the figure answers.
2. **Observation** — the pattern the reader should notice.
3. **Interpretation** — the inference licensed by that pattern.
4. **Inference status** — what is supported, qualified, blocked, or not identified.

A figure should support one principal inference. Separate plots are preferred when trend, distribution, comparison, correlation, and process are all relevant.

## Match the graphic to the claim

| Analytical purpose | Preferred form |
|---|---|
| Change over an ordered or continuous variable | line graph |
| Distribution or frequency | histogram, density, or frequency polygon |
| Discrete comparison | dot plot or bar chart |
| Association between measured variables | scatter plot |
| Mechanism, workflow, or decision sequence | schematic or flow diagram |
| Exact values or heterogeneous units | table |

Axes should put the independent variable horizontally and the dependent variable vertically unless a different orientation materially improves reading.

## Construction rules

- Use a conclusion-oriented title rather than a topic label.
- Label curves, points, and regions directly where practical; do not make the reader shuttle to a legend.
- Use color as a secondary cue, never the only distinction. Preserve line style, marker, position, or text labels in grayscale.
- Keep scales paired when panels are intended for comparison. Declare broken, logarithmic, inverted, or truncated scales next to the axis and in the caption.
- Show uncertainty whenever the plotted quantity is estimated. Match interval precision to estimate precision.
- Avoid unnecessary significant figures, dense background grids, decorative borders, and unexplained abbreviations.
- Place units in axis labels. Spell out nonstandard abbreviations in the figure or caption.
- Keep data marks visually stronger than grids, frames, and annotations.
- Design for final publication size; verify that labels remain legible after reduction.

## Caption template

> **Question.** What does the figure test or compare?  
> **Observation.** What should the reader notice?  
> **Interpretation.** What conclusion follows, under which assumptions?  
> **Inference status.** Supported / qualified / blocked / not identified. State the most important limitation.

## Final verification

Before release, check that plotted values reproduce the generating table or serialized result; axes and units are correct; legends and labels agree with the marks; uncertainty and censoring/saturation are visible; and the figure does not imply a stronger claim than the underlying result object.

## Repository-specific semantics

- Every probe-performance figure must separate probe validity from construct identifiability.
- A high AUC may support label separation while BC1 still blocks semantic interpretation; captions must state both facts.
- Model-blind witness graphics should report the all-family partial witness and the per-family localization together.
- Geometry plots must identify whether separation is raw, scale-controlled, within-relation, or between-relation.
- Use an inference-status footer such as: `probe valid; construct unidentified; semantic interpretation blocked`.
