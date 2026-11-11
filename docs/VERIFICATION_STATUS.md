# Reproduction checks

The comment and documentation revision passed the 11 CPU science tests and 17
CLI checks in the recorded Windows CPU environment. Python syntax and executable
tokens match the previous candidate after excluding comments, docstrings, and
layout tokens. Recomputed tables and the archived result files are byte-identical.

The [external-source equivalence record](../manifests/V2_equivalence.json) covers
eight backbones and both NLinear training identities. It records exact CPU
comparisons of initialization, forward outputs, gradients, and Adam updates.
The revision retains those loaders, model sources, and runtime configurations.

The earlier [GPU verification](../manifests/GPU_single_seed_independent_acceptance.json)
covers two integration checks and eight ETTm1 Seed 951 cases across A, B, and C.
The [numeric comparison](../results/reproduction_validation/ETTm1_seed951_DEV_comparison.csv)
is separate from the five-seed paper evidence. This revision was checked on CPU.
The GPU record applies to the earlier source snapshot and covers the listed
cases only.

Resume checks hash source bytes, so use a fresh output directory with this
snapshot. For existing runs, resume with the snapshot that created them.
