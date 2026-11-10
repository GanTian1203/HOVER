# HOVER

HOVER is a residual adapter for frozen multivariate time-series forecasters.
A shared history encoder feeds a history residual path (HRP) and a
forecast-conditioned refinement path (FCRP). The two paths predict corrections
in history-normalized coordinates and add them to the backbone forecast after
rescaling. Zero-initialized residual heads start training from the frozen forecast.

## Installation

Use Python 3.8.20 for the recorded environment. Create a virtual environment
with `python -m venv .venv`, then activate it with
`.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` on Linux.

```bash
python -m pip install -r requirements-cpu.txt
python -m pip install -e .
```

See the [environment guide](docs/ENVIRONMENTS.md) for the CUDA dependencies and
the environments used for validation.

## Quick start

```bash
python -m hover demo --output outputs/demo.json
```

The demo uses synthetic inputs and runs on CPU without an external backbone or
dataset. To inspect a single paper run, use the experiment commands below.

## Prepare models and data

All eight forecasting backbones are external. The [model registry](docs/EXTERNAL_MODEL_REGISTRY.md)
lists each paper, official repository, pinned commit, source hashes, and license.
For example, obtain and verify the shared NLinear/DLinear source tree with

```bash
python -m hover sources --model NLINEAR --output external_models
python -m hover sources --model NLINEAR --output external_models --verify
```

Use the [backbone guide](docs/EXTERNAL_MODEL_GUIDE.md) for the other models and
local registration. OLinear requires a separately obtained pinned source tree
and a documented basis for your use. The loader checks the source identity.

```bash
python -m hover download --dataset ETTm1 --output data/raw/ETTm1.prefix.csv
python -m hover prepare --dataset ETTm1 --source data/raw/ETTm1.prefix.csv --output data/prepared/ETTm1.npz
```

The download command retrieves a verified TRAIN/DEV prefix for ETTm1. Use
authorized local prefixes for datasets without a verified download boundary.
See [data acquisition](docs/DATA_ACQUISITION.md) and the
[data protocol](docs/DATA_PROTOCOL.md) for sources, splits, and scaling.

## Reproduce the experiments

| Stage | Experiment |
| --- | --- |
| `A` | Ten forecasting systems on 12 tasks |
| `Table2` | Frozen, Simple, and HOVER on five backbones and three datasets |
| `B` | Four mechanism controls on 12 tasks, paired with full HOVER |
| `C` | Frozen OLinear, Simple, and HOVER across 16 dataset–horizon combinations |

The formal runs use seeds 951–955. Strong NLinear in the main table and the
paired NLinear used for HOVER and its controls have separate training recipes.
Each horizon uses a matching output head and frozen checkpoint. The
[experiment guide](docs/PAPER_CODE_RESULTS_MAP.md) links the recipes and results.

The commands below print run plans.

```bash
python -m hover run --stage A --dataset ETTm1 --system NLINEAR_HOVER --seed 951
python -m hover run --stage B --dataset ETTm1 --system history_only --seed 951
python -m hover run --stage C --dataset ETTm1 --horizon 192 --system HOVER --seed 951
python -m hover run --stage Table2 --dataset ECL --system DLINEAR+HOVER --seed 951
```

Use `python -m hover run --stage A` without filters for the full matrix, and
substitute `B`, `C`, or `Table2` for the other matrices. To run training, provide
prepared data, the required external source trees, a device, and an output directory.

```bash
python -m hover run --stage A --dataset ETTm1 --system NLINEAR_HOVER --seed 951 --execute --data-root data/prepared --external-root external_models --device cuda:0 --output outputs/reproduction
```

Use a fresh output directory when switching source snapshots. Resume checks
include source file hashes, which also change when comments are edited.

## Results

All reference metrics are from the development split (DEV). The repository
includes [per-seed results](results/reference), [paper tables](results/tables),
and data for [Figure 1(d)](results/figure1d) and the
[horizon plots](results/horizon_final). The main cross-backbone and horizon
views compare Frozen and HOVER. The archived Simple arms support capacity controls.

```bash
python -m hover.paper_tables --output outputs/paper_tables
python -m hover.presentation --panel Table2 --export-records outputs/Table2_main.json
python -m hover.presentation --panel C --export-records outputs/C_main.json
python -m hover summarize --records outputs/reproduction/records --stage A --output outputs/new_table
python -m hover plot --table outputs/paper_tables/Table1_MSE_MAE_mean_SD.csv --dataset ECL --output outputs/ECL.png
```

For the CPU regression suite, register NLinear and FITS first, then run
`python -B tests/test_science.py`. See the
[verification notes](docs/VERIFICATION_STATUS.md) for the scope of CPU and GPU checks.

## Citation and licensing

**Citation requested for academic and commercial use.** If you use or adapt HOVER in an academic publication, please cite the HOVER paper once its bibliographic details are available, or include the project URL: https://github.com/GanTian1203/HOVER. For commercial use, we similarly request an acknowledgment of HOVER with the paper citation or this repository link in product documentation, technical reports, or other appropriate accompanying materials.

Until the paper's final citation is available, cite this software using [CITATION.cff](CITATION.cff) or provide the repository URL. The paper citation and repository-link request expresses scholarly attribution expectations; **it is not an additional legal requirement of the MIT License**.

The author-owned HOVER source code is provided under the [MIT License](LICENSE), Copyright (c) 2026 GanTian1203. When distributing copies or substantial portions, retain the copyright and permission notice required by MIT. Third-party backbone source code is not distributed here and remains subject to its original terms. See the [third-party notices](THIRD_PARTY_NOTICES.md) and [license scope](docs/LICENSE_REVIEW.md).
