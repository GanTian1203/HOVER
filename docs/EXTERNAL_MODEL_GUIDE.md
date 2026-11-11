# External backbones

The [model registry](EXTERNAL_MODEL_REGISTRY.md) lists the eight models, papers,
official repositories, pinned commits, and licenses. The source configuration
also records the dependency and license file hashes checked by the loader.

## Fetch and verify

```bash
python -m hover sources --model NLINEAR --output external_models
python -m hover sources --model DLINEAR --output external_models
python -m hover sources --model FITS --output external_models
python -m hover sources --model PATCHTST --output external_models
python -m hover sources --model ITRANSFORMER --output external_models
python -m hover sources --model SIGMA --output external_models
python -m hover sources --model TIMEMIXER --output external_models
python -m hover sources --model FITS --output external_models --verify
```

NLinear and DLinear share the same LTSF source tree. Existing source trees
are verified before reuse. Keep the fetched code in the ignored
`external_models/` directory.

## Register local source

```bash
python -m hover sources --model NLINEAR --local-path /path/to/LTSF-Linear --commit 0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6 --output external_models
python -m hover sources --model OLINEAR --local-path /path/to/OLinear --commit f168e01a3e0e316ad98330b5e77afed1f77b0af5 --permission-basis "YOUR_ACTUAL_DOCUMENTED_USE_BASIS" --output external_models
python -m hover sources --model OLINEAR --output external_models --verify
```

Replace the paths and permission placeholder with the actual values. The
registration writes a local pointer and verifies the pinned files. It also
checks `HEAD` for a Git checkout. With an archive, it records the declared
commit and verifies the file hashes. See [OLinear source and licensing](OLINEAR_SOURCE_STATUS.md)
for its local-only requirement.

## Use a backbone

Pass `--external-root external_models` to `run` or `smoke`. Programmatic calls
can set `HOVER_EXTERNAL_ROOT`. Separate processes isolate upstream module names
across backbone families. The [experiment guide](PAPER_CODE_RESULTS_MAP.md)
describes the distinct Strong and paired NLinear recipes, FITS objectives, and
OLinear horizon and Q identities.
