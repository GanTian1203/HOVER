# Reproducibility

The reported experiments use five random seeds (951–955) and select configurations on the designated development split. Primary measures are mean squared error (MSE) and mean absolute error (MAE).

Full reproduction requires the original dataset splits, each upstream backbone's verified source and configuration, frozen checkpoint identity, HOVER configuration, and the exact evaluation protocol.

This documentation will be completed with executable commands and machine-readable experiment records when the release candidates have been audited. Do not infer that unprovided scripts or checkpoints are included in the repository.

## Evidence policy

- Published table and figure values must trace to executed experiments.
- Distinguish paired frozen-backbone comparisons from complete-system comparisons.
- Preserve per-seed records and the data-split boundary used for model selection.
- Compute ranks from the source mean values rather than rounded table displays.
- Do not substitute placeholder values for missing experiment output.
