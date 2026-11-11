# Supplementary materials

## S1 Extended results

The [table index](../PAPER_RESULTS_INDEX.md) gives a direct link for each of Tables I–VI to its manuscript fragment, five-seed means and sample standard deviations, and underlying seed records. [DEV archives](../results/reference) include complete-system, matched-backbone, ablation, and horizon comparisons.

Table VI evaluates Frozen/HOVER pairs on chronological TEST intervals for ECL, Solar, PEMS03, and Exchange with OLinear and paired NLinear. It contains 80 system evaluations and 40 seed pairs. [System scores](test/TEST_system_metrics.csv) · [Paired scores](test/TEST_paired_seeds.csv) · [Means and sample SD](test/TEST_five_seed_summary.csv)

### Recompute published scores

Run `python supplementary/verify_scores.py` to reconcile the six displayed tables with the accepted means and check held-out and PostY seed aggregates.

### Held-out inference

The [inference script](test/test_holdout.py) and [protocol](test/test_protocol.json) reproduce a selected frozen pair using identity-checked inputs. Provide authorized local copies of the original checkpoints, TRAIN/DEV cache and raw dataset. OLinear also requires the full-TRAIN Q matrices. Source, cache, checkpoint, scaler and Q hashes are recorded in the protocol. The evaluator uses TEST targets for scoring after checkpoint selection.

```bash
python supplementary/test/test_holdout.py --help
python supplementary/test/test_holdout.py --dataset ECL --backbone NLINEAR_PAIRED --seed 951 --raw data/electricity.txt.gz --train-cache local/ECL_data.npz --base local/ECL_NLinear.pt --adapter local/ECL_HOVER.pt --external-root external_models --output ECL_NLinear_951_TEST.json
```

For OLinear, select `OLINEAR` and supply `--q-history` and `--q-future`. Both loader and scorer paths were checked on two ECL DEV-prefix windows per backbone. The full TEST evidence comes from the original chronological replay.

## S2 Protocol and implementation

The [data protocol](../docs/DATA_PROTOCOL.md) defines TRAIN, inner fitting and selection, DEV, time features, and scaling. PostY fitting uses the matched training configuration documented above. The published PostY package provides records and settings for that comparison. The TEST replay uses fixed checkpoints and TRAIN scaling. Its records give the evaluation bounds and source and checkpoint identities.

Paper experiments use seeds 951–955 and a history of 336 steps. Tables I–IV use a 96-step forecast. Table V uses [horizon-specific recipes](../src/hover/configuration/horizons.json). The [native recipes](../src/hover/configuration/native.json), [paired NLinear recipes](../src/hover/configuration/paired_nlinear.json), and [inner-training recipes](../src/hover/configuration/inner_native.json) specify objectives and selection. [Execution commands](../docs/PAPER_CODE_RESULTS_MAP.md) · [External source registry](../docs/EXTERNAL_MODEL_REGISTRY.md)

## S3 Structural controls

Table IV compares Full HOVER with four retrained controls on paired NLinear. Only HRP omits the conditioned path. No HRP output removes the direct residual and keeps the shared encoder. Additive fusion replaces the projected product with projected addition. History-surrogate supplies an observed-history surrogate as the condition. The records use `concatenation` and `history_condition_replacement` for the last two controls.

[Seed scores](../results/reference/ablation_DEV_seeds.csv) · [Means and sample SD](../results/tables/Table3_MSE_MAE_mean_SD.csv) · [Paired differences](../results/tables/Table3_all_control_pairs.csv) · [Figure 4](../assets/figure4.pdf)

## S4 Comparison with correction methods

PostY denotes the official offline multiplicative Ada-Y variant. The DLinear comparisons pair the same checkpoint, targets, and seed. The PostY state is selected by DEV MSE. Per-run trainable parameter counts appear in the DEV records. HOVER has 42,464 trainable parameters in this configuration. [Pinned upstream](https://github.com/Anoise/Adapter/tree/0add06ea7b4d2e0a84c364a8be72eef2676a92f2) · [Settings and hashes](posty/protocol.json)

| Evaluation | Coverage | Evidence |
| --- | --- | --- |
| DEV | ECL and PEMS03, Frozen, PostY and HOVER, five seeds | [Seed scores](posty/DEV_seed_results.csv) · [Means and sample SD](posty/DEV_five_seed_summary.csv) |
| DLinear TEST replay | ECL and PEMS03, Frozen, Simple, PostY and HOVER, five seeds | [Seed scores](posty/FIG7_DLINEAR_TEST_RESULTS.csv) · [Means and sample SD](posty/FIG7_DLINEAR_TEST_FIVE_SEED_SUMMARY.csv) |

PostY fitting uses the matched training configuration documented above. The published PostY package provides records and settings for that comparison. The TEST replay uses fixed checkpoints on the chronological holdout. Table VI reports the separate OLinear and paired-NLinear evaluation.

## S5 Trajectory and local behavior

[Figure 6](../assets/figure6.pdf) shows ECL with OLinear and PEMS03 with paired NLinear. [Figure 7](../assets/figure7.pdf) compares four DLinear methods on identical DEV windows at seed 951. The typical and illustrative cases use different selection rules. Figure 7(c) shows a window with lower MSE for Simple.

- [Figure 6 case identities](trajectories/figure6_source_manifest.json) · [Figure 7 case identities](trajectories/selected_cases.json) · [Selection summary](trajectories/FIG7_CANDIDATE_SELECTION.csv) · [Local views](trajectories/Figure7_Local_Views.pdf)
- [Window and segment errors](trajectories/FIG7_WINDOW_SEGMENT_SUMMARY.csv) · [Five-seed segment summaries](trajectories/FIG7_FIVE_SEED_SEGMENT_SUMMARY.csv) · [Case seed scores](trajectories/FIG7_SEED_CONSISTENCY.csv)
- [Figure 6 forecast values](trajectories/figure6_selected_forecasts.csv) · [Figure 7 per-step forecasts and normalized branches](trajectories/figure7_selected_forecasts.csv)
- [Branch reconstruction](trajectories/FIG7_BRANCH_RECONSTRUCTION.json) · [Branch segment features](trajectories/FIG7_BRANCH_SEGMENT_FEATURES.csv)

For 20 case–seed pairs, the normalized branches reconstruct the HOVER forecast after rescaling their sum and adding the frozen forecast. Their magnitudes describe the learned decomposition. Overlapping channel–origin windows support descriptive statistics. The five seeds are the units for run-to-run variability.

[Rebuild Figures 6 and 7](trajectories/README.md) from the pinned upstream observations and published forecasts, or rerun inference on the six DEV cases with the reference checkpoints.

## S6 Additional analysis

[DEV task scores](trajectories/FIG7_DEV_SYSTEM_RESULTS.csv), [morphology summaries](trajectories/FIG7_MORPHOLOGY_SUMMARY.csv), and [five-seed morphology summaries](trajectories/FIG7_MORPHOLOGY_FIVE_SEED_SUMMARY.csv) extend the selected windows. The [horizon archive](../results/horizon_final/DEV_48_system_groups.csv) includes the Simple arms. [Source manifest](../manifests/paper_supplement_sources.json)
