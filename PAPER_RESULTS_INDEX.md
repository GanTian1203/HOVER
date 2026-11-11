# Paper results and reproduction

Numbering follows the [manuscript](paper/HOVER.pdf). The LaTeX fragments
contain the displayed values. The extensions report means and sample standard
deviations across seeds 951–955. The underlying seed records include
available source and checkpoint identifiers.

| Paper table | Displayed table | Five-seed mean and sample SD | Seed records |
| --- | --- | --- | --- |
| I · Complete systems · DEV | [Table I](supplementary/paper-tables/table1.tex) | [120 system groups](results/tables/Table1_MSE_MAE_mean_SD.csv) | [600 records](results/reference/main_DEV_seeds.csv) |
| II · Matched Frozen/HOVER · DEV | [Table II](supplementary/paper-tables/table2.tex) | [Three-arm extension](results/tables/Table2_three_arm_MSE_MAE_mean_SD.csv) | [225 records](results/reference/cross_backbone_DEV_records.json) |
| III · Capacity control · DEV | [Table III](supplementary/paper-tables/table3.tex) | [Same archive, paired NLinear and DLinear](results/tables/Table2_three_arm_MSE_MAE_mean_SD.csv) | [Three-arm records](results/reference/cross_backbone_DEV_records.json) |
| IV · Structural controls · DEV | [Table IV](supplementary/paper-tables/table4.tex) | [Full HOVER and four controls](results/tables/Table3_MSE_MAE_mean_SD.csv) | [300 records](results/reference/ablation_DEV_seeds.csv) |
| V · Prediction horizons · DEV | [Table V](supplementary/paper-tables/table5.tex) | [48 three-arm system groups](results/horizon_final/DEV_48_system_groups.csv) | [240 records](results/reference/horizon_accepted_DEV_records.json) |
| VI · Chronological TEST | [Table VI](supplementary/paper-tables/table_test.tex) | [Eight matched groups](supplementary/test/TEST_five_seed_summary.csv) | [80 system scores](supplementary/test/TEST_system_metrics.csv) · [40 pairs](supplementary/test/TEST_paired_seeds.csv) |

For Table II use FROZEN and HOVER rows in the extension. For Table III use
the FROZEN, SIMPLE and HOVER rows for NLINEAR_PAIRED and DLINEAR on ECL,
Solar and PEMS03. For Table V use FROZEN and HOVER at each horizon. Table I
uses standalone Strong NLinear and an NLinear+HOVER system built on paired
NLinear. The matched tables use the paired recipe for NLinear. See the
[source manifest](manifests/paper_supplement_sources.json) for the released
file hashes and the [supplement](supplementary/README.md) for evaluation
protocols and additional scores.

| Figure | Evidence and reproduction |
| --- | --- |
| [1](assets/figure1.pdf) · Motivation | Schematic illustration |
| [2](assets/figure2.pdf) · Architecture | [Method implementation](src/hover/method.py) |
| [3](assets/figure3.pdf) · Cross-backbone and capacity | [Paired gain records](results/tables/Table2_all_same_backbone_pairs.csv) and the Tables II–III seed archive above |
| [4](assets/figure4.pdf) · Mechanism controls | [Paired differences](results/tables/Table3_all_control_pairs.csv) |
| [5](assets/figure5.pdf) · Horizons | [Horizon results](results/horizon_final) |
| [6](assets/figure6.pdf) and [7](assets/figure7.pdf) · DEV waveforms | [Redraw or rerun selected inference](supplementary/trajectories/README.md) |

[PostY and four-method DLinear TEST evidence](supplementary/README.md#s4-comparison-with-correction-methods)
includes the scope, seed records and [pinned PostY protocol](supplementary/posty/protocol.json).
See the [experiment guide](docs/PAPER_CODE_RESULTS_MAP.md) for plan and execution modes.
