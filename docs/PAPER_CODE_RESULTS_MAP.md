# Experiments and results

`python -m hover run` plans or executes the paper experiments.
`python -m hover.presentation --panel <name>` selects the corresponding
reference records for a paper panel.

| Panel | Paper comparison | Reference records |
| --- | --- | --- |
| `A` | Ten systems on 12 tasks | [600 seed records](../results/reference/main_DEV_seeds.csv) |
| `Table2` | Frozen and HOVER on five backbones and ECL, Solar, and PEMS03 | 150 paper records from the [225-record three-arm archive](../results/reference/cross_backbone_DEV_records.json) |
| `Capacity` | Frozen, Simple, and HOVER on paired NLinear and DLinear on the same three datasets | 90 records selected from the three-arm archive |
| `B` | Full HOVER and four mechanism controls on paired NLinear across 12 tasks | [240 control and 60 full-HOVER records](../results/reference/ablation_DEV_seeds.csv) |
| `C` | Frozen OLinear and HOVER across four horizons per dataset | [160 paper records](../results/presentation/C_main_accepted_DEV_records.json) from the [240-record three-arm archive](../results/reference/horizon_accepted_DEV_records.json) |

Each formal result uses seeds 951–955. The main table uses Strong NLinear as a
standalone baseline and P3 paired NLinear for NLinear+HOVER. The latter also
supplies the denominator for the same-seed gains in Figure 1(d). The
[Figure 1(d) data](../results/figure1d) cover ECL, Solar, and PEMS03. The
separately rendered R15 figure uses these values.

## Recipes

- [Native recipes](../src/hover/configuration/native.json) retain model-specific
  training schedules and objectives, including the FITS reconstruction objective.
- [Paired NLinear recipes](../src/hover/configuration/paired_nlinear.json) define
  the backbone and adapter used for the NLinear controls.
- [Horizon recipes](../src/hover/configuration/horizons.json) define the forecast
  lengths. Each length uses its own output head and frozen checkpoint. OLinear
  Q transforms are tied to both the horizon and the TRAIN or INNER_FIT split.
- [Inner-training recipes](../src/hover/configuration/inner_native.json) support
  adapter selection. Seed 731 selects on INNER_VAL, and seed 732 confirms the
  selection. Formal checkpoint selection follows the experiment's DEV protocol.

The mechanism control keyed `concatenation` uses `LayerNorm(GELU(u + v))` on
the projected history and forecast features. The history-surrogate control
uses only observed history. Simple uses a single shared MLP with the same input
information and a similar number of trainable parameters.

## Reporting

[Paper tables](../results/tables) include five-seed means, sample standard
deviations, paired differences, and win counts. [Horizon results](../results/horizon_final)
retain all 16 combinations and the complete three-arm evidence, including the
60 reused horizon-96 records. Report MSE and MAE separately. Comparisons with
Frozen measure adaptation gains, while comparisons with Simple measure gains
over the capacity control. All reference metrics are DEV.
