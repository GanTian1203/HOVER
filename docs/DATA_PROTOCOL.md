# Data provenance, coordinates and roles

Exact URLs and source identities are in [datasets.json](../src/hover/configuration/datasets.json).

| Dataset | Variables | Nominal interval | TRAIN rows | DEV rows |
|---|---:|---|---|---|
| ETTh1 / ETTh2 | 7 | 1 hour | [0,8640) | [8640,11520) |
| ETTm1 / ETTm2 | 7 | 15 minutes | [0,34560) | [34560,46080) |
| ECL | 321 | 1 hour | [0,18412) | [18412,21044) |
| Traffic | 862 | 1 hour | [0,12280) | [12280,14036) |
| Solar | 137 | 10 minutes | [0,36792) | [36792,42048) |
| Exchange | 8 | one recorded day | [0,5311) | [5311,6071) |
| PEMS03 | 358 | 5 minutes | [0,15724) | [15724,20966) |
| PEMS04 | 307 | 5 minutes | [0,10195) | [10195,13593) |
| PEMS07 | 883 | 5 minutes | [0,16934) | [16934,22579) |
| PEMS08 | 170 | 5 minutes | [0,10713) | [10713,14284) |

ETT uses ETDataset commit 1d16c8f4f943005d613b5bc962e9eeb06058cf07.
The old six tasks retain the accepted column order and actual release.
ECL/Traffic/Solar/Exchange use the research-publisher release at
7f402f185cc2435b5e66aed13a3b560ed142e023. ECL is the processed 321-variable
benchmark. It is not byte-certified against a differently packaged TSLib CSV.
PEMS sources use ASTGNN commit 9c2e19b98c4cedf1f35214d8789685b6381b3aad.

Original full CSV hashes were not retained for ETTh1/2. Exact URLs/commit,
headers and canonical float32 normalized-prefix hashes are retained instead.
Solar/Exchange preserve decoded-prefix byte hashes; ECL/Traffic preserve
compressed-file hashes. ETTm/PEMS have publisher byte hashes. Cache serialization
can differ, so source, canonical numeric and cache-file identities are separate.

All original columns remain in original order. PEMS selects flow0 and treats
zeros as valid. The admitted prefixes have finite labels throughout.
Unexpected nonfinite targets cause rejection; no target imputation occurs.

Formal mean and population SD use full TRAIN only. An origin is the first
future target. TRAIN origins are 336 through train_end-H, DEV origins train_end
through dev_end-H, inclusive. A DEV history may include TRAIN observations;
all future labels remain within DEV.

Nested selection uses floor(0.8*TRAIN) for INNER_FIT and the remaining TRAIN
for INNER_VAL. Raw-prefix caches use INNER_FIT raw statistics. Legacy caches
restandardize their accepted affine z coordinates on INNER_FIT; float32 rounding
is retained. Official DEV is not decoded for inner selection. Inner bases use
fixed final epochs without INNER_VAL checkpoint selection.

OLinear Q binds role, dataset, cache, history, horizon, statistic rows and file
hashes. Near-constant channels are skipped only for covariance estimation;
prediction channels are complete. Formal Q affine-decodes the accepted z
prefix and casts to float32. Inner Q prefers raw values if available, otherwise
uses the same decoder at an INNER_FIT-only cutoff. Covariance is divided by
its column diagonal, averaged in float32 and passed to the original
lower-triangle eigensolver convention. No identity fallback is introduced.

ETT retains verified calendar features. Other admitted releases have no invented
calendar inputs. SiGMA's frozen official forward ignores ETT calendar arguments;
this behavior is retained. Hourly timeF configurations drop the extra ETTm
minute field as in the original interface.

MSE/MAE use accepted standardized multivariate targets. INNER_VAL values have
their role-specific scale and cannot be directly compared to formal DEV.
Inverse normalization is provided.

ETT metadata records CC BY-ND 4.0. Processed research releases retain their
research-use provenance and underlying-source qualifications. Exchange's
ultimate provider provenance remains limited. PEMS download access does not
itself establish redistribution rights. No raw or processed signals are included.
