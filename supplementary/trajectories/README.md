# Reproduce the waveform figures

Figure 6 uses ECL with frozen OLinear and PEMS03 with paired NLinear. Figure 7 uses Frozen, Simple, offline multiplicative PostY, and HOVER on matched frozen DLinear forecasts. The case manifests fix the zero-based channel, forecast origin, seed 951, and zoom bounds. All cases are DEV. The forecast origin is the first target row.

## Redraw from the published forecasts

Obtain the pinned [ECL benchmark file](https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/7f402f185cc2435b5e66aed13a3b560ed142e023/electricity/electricity.txt.gz) and [PEMS03 benchmark file](https://raw.githubusercontent.com/guoshnBJTU/ASTGNN/9c2e19b98c4cedf1f35214d8789685b6381b3aad/data/PEMS03/PEMS03.npz) from their upstream sources and follow the applicable dataset terms. The [dataset configuration](../../src/hover/configuration/datasets.json) gives file identities, channels and split boundaries. The package distributes forecasts and aggregate metrics. Observations and targets are reconstructed locally.

From the installed repository root, run:

```bash
python supplementary/trajectories/rebuild_waveforms.py --ecl data/raw/electricity.txt.gz --pems03 data/raw/PEMS03.npz --output outputs/waveforms
```

The command checks the raw file hashes, uses TRAIN population statistics and validates the full standardized TRAIN/DEV array identity. It decodes numeric values only through the end of DEV. PEMS03 uses feature 0 in original channel order. The output directory contains both redrawn figures, local views, prepared TRAIN/DEV caches, and a CSV of 84 full-window or 24-step segment MSE/MAE comparisons. Published figures remain under assets. Font and rendering-library differences may affect the output files.

## Rerun inference with the reference checkpoints

Use the same command with a new output directory and `--checkpoints /path/to/local-replay.json`. Copy the [example JSON](checkpoint_paths.example.json), replace the paths with local files, and keep external sources separate. Relative paths are resolved from the JSON directory. The path map requires reference checkpoints whose SHA256 matches the manifests, the full-TRAIN OLinear Q matrices, the separately obtained pinned LTSF and OLinear sources, and the pinned [PostY source](../posty/protocol.json). OLinear source registration requires a documented use basis as described in the [external model guide](../../docs/EXTERNAL_MODEL_GUIDE.md).

This option runs inference on the six selected DEV windows and checks the new outputs against the saved forecasts before plotting. It uses the original model identities and checks shared frozen weights. The public package provides the hashes and runner. The reference weights, Q files, external sources and upstream data are separate inputs. The public redraw above requires only the two upstream datasets.

## Scope of training and held-out reproduction

The A and Table2 runners train the HOVER and Simple arms for their documented settings. This waveform replay requires the reference checkpoint bytes. A new fit can be assessed with the same window protocol, and its predictions can differ numerically. The supplementary records describe the original PostY DEV training configuration and the four-method DLinear TEST evaluation. Standalone PostY fitting and full DLinear TEST evaluation remain outside the released runner. The separate Table VI evaluator covers OLinear and paired NLinear. The waveform command uses DEV only.
