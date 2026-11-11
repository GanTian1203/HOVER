"""Prepare accepted TRAIN/DEV prefixes without decoding protected suffix values."""
import csv
import gzip
import hashlib
import re
from pathlib import Path
import urllib.request
import zipfile

import numpy as np

from .data import npy_prefix
from .io import Blocked, config, save, sha256


def prepare(dataset, source, destination):
    definition = config("datasets")[dataset]
    source, destination = Path(source), Path(destination)
    if destination.exists() or destination.with_suffix(".json").exists():
        raise FileExistsError("Prepared prefix already exists")
    expected = definition.get("source_full_SHA256")
    source_digest = sha256(source)
    is_prefix = bool(definition.get("source_prefix_SHA256") and source_digest == definition["source_prefix_SHA256"])
    if expected and source_digest != expected and not is_prefix:
        raise ValueError("Published source byte identity mismatch")
    stop, train, channels = definition["DEV_rows"][1], definition["TRAIN_rows"][1], definition["channels"]
    dates = None
    prefix_hash = None
    if definition["format"] == "ett_csv":
        rows, stamps = [], []
        with source.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            if next(reader) != definition["header"]:
                raise ValueError("ETT column order mismatch")
            for _ in range(stop):
                row = next(reader)
                stamps.append(row[0])
                rows.append([float(v) for v in row[1:]])
        raw = np.asarray(rows, dtype=np.float64)
        dtype = "datetime64[m]" if dataset.startswith("ETTm") else "datetime64[h]"
        dates = np.asarray(stamps, dtype=dtype)
        if not (np.diff(dates) == np.timedelta64(definition["nominal_minutes"], "m")).all():
            raise ValueError("ETT calendar interval mismatch")
    elif definition["format"] == "gzip_csv":
        rows, digest = [], hashlib.sha256()
        with gzip.open(source, "rb") as stream:
            for _ in range(stop):
                line = stream.readline()
                if not line:
                    raise ValueError("Truncated source prefix")
                digest.update(line)
                rows.append(np.fromstring(line.decode("ascii").strip(), sep=",", dtype=np.float64))
        raw = np.stack(rows)
        prefix_hash = digest.hexdigest()
        if definition.get("decoded_prefix_SHA256") and definition["decoded_prefix_SHA256"] != prefix_hash:
            raise ValueError("Exact decoded prefix identity mismatch")
    elif definition["format"] == "pems_npz":
        with zipfile.ZipFile(source) as archive:
            raw = npy_prefix(archive, "data", stop, channels)
        if raw.ndim != 3 or raw.shape[2] < 1:
            raise ValueError("PEMS flow0 source schema mismatch")
        raw = raw[:, :, 0].astype(np.float64)
    else:
        raise Blocked("No admitted prefix reader")
    if raw.shape != (stop, channels) or not np.isfinite(raw).all():
        raise ValueError("Accepted all-finite target mask changed; do not impute labels")
    mu, sd = raw[:train].mean(0), raw[:train].std(0)
    sd = np.where(sd < 1e-12, 1., sd)
    arrays = dict(z=((raw - mu) / sd).astype(np.float32), mu=mu, sd=sd)
    z_identity = hashlib.sha256(arrays["z"].tobytes()).hexdigest()
    if definition.get("canonical_z_SHA256") and z_identity != definition["canonical_z_SHA256"]:
        raise ValueError("Canonical standardized prefix differs from accepted ETTh evidence")
    if definition["keep_raw"]:
        arrays.update(raw=raw, observed=np.isfinite(raw))
    if dates is not None:
        arrays["dates"] = dates
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    result = dict(status="DONE", dataset=dataset, TRAIN_rows=definition["TRAIN_rows"], DEV_rows=definition["DEV_rows"],
        cache_SHA256=sha256(destination), accepted_cache_SHA256=definition["accepted_cache_SHA256"],
        source_SHA256=source_digest, source_identity_kind="LEGAL_TRAIN_DEV_PREFIX" if is_prefix else "PUBLISHED_SOURCE",
        pinned_full_source_SHA256=expected, decoded_prefix_SHA256=prefix_hash,
        canonical_z_SHA256=z_identity,
        TEST_access=False, rows_decoded=stop, channel_order=definition["column_order"],
        cache_byte_equivalence=sha256(destination) == definition["accepted_cache_SHA256"])
    save(destination.with_suffix(".json"), result)
    return result


def download(dataset, destination):
    definition = config("datasets")[dataset]
    destination = Path(destination)
    length = definition.get("source_prefix_bytes")
    expected = definition.get("source_prefix_SHA256")
    if not length or not expected or definition["format"] != "ett_csv":
        raise Blocked("No audited byte-bounded download is available for this dataset; provide a lawful local TRAIN/DEV prefix. Whole-source transport is disabled.")
    if destination.exists():
        if destination.stat().st_size != length or sha256(destination) != expected:
            raise FileExistsError("Existing source is not the admitted prefix; it is preserved")
        return dict(status="DONE",reused=True,dataset=dataset,source_SHA256=expected,
                    source_identity_kind="LEGAL_TRAIN_DEV_PREFIX",downloaded_bytes=0,TEST_bytes_requested=0,TEST_access=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(definition["source_url"], headers={"Range": "bytes=0-"+str(length-1), "Accept-Encoding": "identity"})
    with urllib.request.urlopen(request, timeout=60) as response:
        match = re.fullmatch(r"bytes 0-(\d+)/(\d+|\*)", response.headers.get("Content-Range", ""))
        if response.status != 206 or not match or int(match.group(1)) != length-1:
            raise Blocked("Publisher did not honor the exact TRAIN/DEV byte range; body was not read. No full-file fallback is permitted.")
        remaining = length
        with destination.open("xb") as stream:
            while remaining:
                block = response.read(min(65536, remaining))
                if not block:
                    raise ValueError("Truncated bounded download")
                stream.write(block);remaining -= len(block)
    if sha256(destination) != expected:
        raise ValueError("Source prefix SHA mismatch; preserve file for diagnosis")
    result = dict(status="DONE", dataset=dataset, source_SHA256=sha256(destination), source_identity_kind="LEGAL_TRAIN_DEV_PREFIX",
                  downloaded_bytes=length, numeric_values_decoded=0, TEST_bytes_requested=0, TEST_access=False)
    save(destination.with_suffix(destination.suffix+".download.json"), result)
    return result
