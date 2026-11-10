# Data acquisition

The download command supports the verified ETTm1 TRAIN/DEV prefix. It requests
bytes 0–6846441 from the pinned source and requires HTTP 206 with the exact
`Content-Range`. It validates the prefix SHA256 and rejects responses that
would require reading beyond the range. The prefix contains the header and
46,080 rows. Its identity is separate from the full-source hash.

```bash
python -m hover download --dataset ETTm1 --output data/raw/ETTm1.prefix.csv
python -m hover prepare --dataset ETTm1 --source data/raw/ETTm1.prefix.csv --output data/prepared/ETTm1.npz
```

For other datasets, prepare an authorized local input that supplies the
TRAIN/DEV rows and meets the source checks in
[datasets.json](../src/hover/configuration/datasets.json). A compressed source
may require a publisher-provided prefix. Use the original column order and
follow the publisher's terms. [Data protocol](DATA_PROTOCOL.md) gives the exact
splits, scaling rules, and time features. The
[ETTm1 verification record](../manifests/phase3_real_data_verification.json)
documents the verified prefix and preprocessing comparison.
