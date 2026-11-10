# External forecasting backbones

HOVER is released separately from the forecasting models used in experiments.

Backbone source code is not copied into this repository. For each supported model, the reproducibility documentation will identify:

- The original paper and upstream repository.
- The tested upstream revision (commit) and source-file checksums where available.
- The applicable license and any limitations on redistribution.
- The installation method or a local-source-path integration option.
- The frozen-checkpoint and configuration identity used in an experiment.

The evaluation includes backbones such as NLinear, DLinear, FITS, PatchTST, iTransformer, SiGMA, TimeMixer, and OLinear. Their exact tested revisions and licensing details must be verified against the experiment records before publication.

In particular, do not add OLinear or any other upstream source to this repository without confirming redistribution rights.
