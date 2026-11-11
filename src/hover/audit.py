"""Independent checkpoint identity and numeric aggregate verification."""
from pathlib import Path

import numpy as np
import torch

from .data import Windows
from .io import identity, load, sha256


def audit_fit(directory, data_root=None, base_directory=None):
    directory = Path(directory)
    receipt = load(directory / "receipt.json")
    checkpoint = directory / "best.pt"
    if receipt["status"] != "DONE" or receipt["TEST_access"] is not False:
        raise ValueError("Unaccepted fit")
    if sha256(checkpoint) != receipt["checkpoint_SHA256"]:
        raise ValueError("Checkpoint hash mismatch")
    saved = torch.load(checkpoint, map_location="cpu")
    definition = {k: receipt[k] for k in ("data", "protocol", "seed", "binding", "base_identity", "method")}
    if saved["definition"] != definition or identity(definition) != receipt["fit_id"] or saved["fit_id"] != receipt["fit_id"]:
        raise ValueError("Checkpoint role/configuration/seed/model identity mismatch")
    if saved["best_epoch"] != receipt["best_epoch"]:
        raise ValueError("Checkpoint epoch mismatch")
    if receipt["base_identity"]:
        if base_directory is None:
            raise ValueError("Independent adapter audit requires --base-fit-dir")
        base_directory = Path(base_directory)
        base = load(base_directory / "receipt.json")
        if base["checkpoint_SHA256"] != receipt["base_identity"]["checkpoint_SHA256"] or sha256(base_directory / "best.pt") != base["checkpoint_SHA256"]:
            raise ValueError("Frozen native checkpoint identity mismatch")
        native = torch.load(base_directory / "best.pt", map_location="cpu")["state"]
        prefix = "base.network." if receipt["binding"].get("arm") == "HOVER" else "base."
        frozen = {k[len(prefix):]: v for k, v in saved["state"].items() if k.startswith(prefix)}
        if set(frozen) != set(native) or not all(torch.equal(frozen[k], native[k]) for k in native):
            raise ValueError("Adapter checkpoint changed frozen weights")
    if receipt["metrics"] is None:
        if receipt["data"]["role"] != "inner" or receipt["protocol"]["selection"] != "fixed_last":
            raise ValueError("Unexpected missing validation metrics")
        return dict(status="PASS", kind="INNER_FIT_FIXED_LAST_IDENTITY", fit_id=receipt["fit_id"], TEST_access=False)
    with np.load(directory / "DEV_or_INNER_VAL_aggregates.npz", allow_pickle=False) as arrays:
        sse, sae, count, origins = (arrays[k] for k in ("SSE", "SAE", "valid_count", "origins"))
        if not np.isfinite(sse).all() or not np.isfinite(sae).all() or (count <= 0).any():
            raise ValueError("Invalid aggregate values")
        if not (count == receipt["data"]["channels"] * receipt["data"]["horizon"]).all():
            raise ValueError("Aggregate mask mismatch")
        meta = receipt["data"]
        expected_origins = np.arange(meta["statistics_rows"][1], meta["stop"] - meta["horizon"] + 1)
        if not np.array_equal(origins, expected_origins):
            raise ValueError("Aggregate origin identity mismatch")
        metrics = dict(mse=float(sum(sse) / sum(count)), mae=float(sum(sae) / sum(count)), valid_count=int(sum(count)))
        for name in ("mse", "mae"):
            if abs(metrics[name] - receipt["metrics"][name]) > 1e-10:
                raise ValueError("Recomputed metric mismatch")
        if metrics["valid_count"] != receipt["metrics"]["valid_count"]:
            raise ValueError("Aggregate element count mismatch")
        if data_root:
            meta = receipt["data"]
            data = Windows(meta["dataset"], Path(data_root) / (meta["dataset"] + ".npz"), meta["horizon"], meta["role"])
            if data.manifest() != meta or not np.array_equal(origins, data.origins["DEV"]):
                raise ValueError("Evaluation origin identity mismatch")
    return dict(status="PASS", kind="NUMERIC_AGGREGATE_AND_CHECKPOINT_AUDIT", fit_id=receipt["fit_id"], metrics=metrics, TEST_access=False)
