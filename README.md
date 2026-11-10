# HOVER

**History- and Output-Conditioned Variable-Shared Error Refinement for Frozen Time-Series Forecasters**

HOVER is a residual refinement module for trained multivariate time-series forecasters. It learns corrections from recent observations and the current base forecast while keeping the forecasting backbone frozen.

The method uses a shared historical encoder and two complementary paths:

- **History Residual Path (HRP):** estimates a direct correction from observed history.
- **Forecast-Conditioned Refinement Path (FCRP):** conditions an additional residual on the base forecast's change relative to the last observation.

The correction parameters are shared across variables.

## Code and reproducibility

The implementation, verified training and evaluation scripts, experiment configurations, and reproducibility records will be included after final source and result verification.

This repository does not redistribute third-party forecasting backbone source code. Backbone integrations use separately obtained upstream implementations or a user-provided local path. See [External backbones](docs/external_backbones.md).

## Reproduction protocol

The experimental protocol and evidence requirements are described in [Reproducibility](docs/reproducibility.md). Release scripts and configurations will be linked to their corresponding verified experiment records.

## Citation

Paper metadata and a BibTeX citation will be added when the bibliographic record is finalized.
