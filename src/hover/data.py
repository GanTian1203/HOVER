"""Window sampling, scaling, and OLinear transforms for TRAIN and DEV."""
import hashlib
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import torch

from .io import Blocked, config, identity, load, save, sha256


def npy_prefix(archive, key, rows, channels=None):
    """Read the first rows of a numeric NPY member without decoding its suffix."""
    with archive.open(key + ".npy") as stream:
        version = np.lib.format.read_magic(stream)
        if version not in ((1, 0), (2, 0)):
            raise ValueError("Unreviewed NPY header")
        reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = reader(stream)
        if fortran or dtype.hasobject or not (0 < rows <= shape[0]):
            raise ValueError("Invalid prefix layout")
        if channels is not None and shape[1] != channels:
            raise ValueError("Dataset channel mismatch")
        count = rows * int(np.prod(shape[1:]))
        blob = stream.read(count * dtype.itemsize)
        if len(blob) != count * dtype.itemsize:
            raise ValueError("Truncated numeric prefix")
        return np.frombuffer(blob, dtype=dtype, count=count).reshape((rows,) + shape[1:]).copy()


def time_features(dates, minute=False):
    stamp = pd.DatetimeIndex(dates)
    arrays = [stamp.hour.to_numpy() / 23 - .5, stamp.dayofweek.to_numpy() / 6 - .5,
              (stamp.day.to_numpy() - 1) / 30 - .5, (stamp.dayofyear.to_numpy() - 1) / 365 - .5]
    if minute:
        arrays = [stamp.minute.to_numpy() / 59 - .5] + arrays
    return np.stack(arrays, 1).astype(np.float32)


class Windows:
    """Sample history and target windows within the selected data split."""

    def __init__(self, dataset, cache, horizon=96, role="formal", device="cpu", definitions=None):
        definitions = config("datasets") if definitions is None else definitions
        if dataset not in definitions or role not in ("formal", "inner"):
            raise ValueError("Dataset/role not admitted")
        if horizon < 1:
            raise ValueError("Invalid horizon")
        self.dataset, self.horizon, self.role, self.device = dataset, horizon, role, device
        self.definition = definitions[dataset]
        self.channels = self.definition["channels"]
        self.full_train_end = self.definition["TRAIN_rows"][1]
        self.train_end = self.full_train_end if role == "formal" else int(.8 * self.full_train_end)
        self.stop = self.definition["DEV_rows"][1] if role == "formal" else self.full_train_end
        cache = Path(cache)
        metadata = load(cache.with_suffix(".json"))
        if metadata["dataset"] != dataset or metadata["cache_SHA256"] != sha256(cache):
            raise ValueError("Dataset/cache identity mismatch")
        if metadata["TRAIN_rows"] != self.definition["TRAIN_rows"] or metadata["DEV_rows"] != self.definition["DEV_rows"]:
            raise ValueError("Dataset role boundaries changed")
        if metadata["TEST_access"] is not False:
            raise ValueError("Unaccepted data access")
        source_expected = self.definition.get("source_full_SHA256")
        prefix_ok = (metadata.get("source_identity_kind") == "LEGAL_TRAIN_DEV_PREFIX" and
                     self.definition.get("source_prefix_SHA256") is not None and
                     metadata.get("source_SHA256") == self.definition["source_prefix_SHA256"])
        if source_expected and metadata.get("source_SHA256") != source_expected and not prefix_ok:
            raise ValueError("Prepared data does not match the pinned public source")
        self.cache_SHA256 = metadata["cache_SHA256"]
        with zipfile.ZipFile(cache) as archive:
            z = npy_prefix(archive, "z", self.stop, self.channels)
            with archive.open("mu.npy") as f:
                mu = np.lib.format.read_array(f, allow_pickle=False)
            with archive.open("sd.npy") as f:
                sd = np.lib.format.read_array(f, allow_pickle=False)
            dates = npy_prefix(archive, "dates", self.stop) if "dates.npy" in archive.namelist() else None
            raw_key = "raw" if "raw.npy" in archive.namelist() else "z"
            values = npy_prefix(archive, raw_key, self.stop, self.channels) if role == "inner" else None
        if z.dtype != np.float32 or z.shape != (self.stop, self.channels) or not np.isfinite(z).all():
            raise ValueError("Input cache shape/mask mismatch")
        if role == "formal" and self.definition.get("canonical_z_SHA256"):
            if hashlib.sha256(z.tobytes()).hexdigest() != self.definition["canonical_z_SHA256"]:
                raise ValueError("Canonical ETTh prefix mismatch")
        if mu.shape != sd.shape or mu.shape != (self.channels,) or not np.isfinite(mu).all() or not np.isfinite(sd).all() or not (sd > 0).all():
            raise ValueError("Invalid scaler")
        self.coordinate_policy = "accepted fullTRAIN standardized float32"
        if role == "inner":
            # Inner validation uses statistics fitted only on INNER_FIT.
            values = values.astype(np.float64)
            if not np.isfinite(values).all():
                raise ValueError("INNER_FIT/INNER_VAL mask mismatch")
            mu = values[:self.train_end].mean(0)
            sd = values[:self.train_end].std(0)
            sd = np.where(sd < 1e-12, 1., sd)
            z = ((values - mu) / sd).astype(np.float32)
            self.coordinate_policy = "raw INNER_FIT statistics" if raw_key == "raw" else "accepted affine z restandardized on INNER_FIT; rounding retained"
        self.mu, self.sd, self.z = mu, sd, torch.tensor(z)
        self.origins = {"TRAIN": np.arange(336, self.train_end - horizon + 1),
                        "DEV": np.arange(self.train_end, self.stop - horizon + 1)}
        if not all(len(v) > 0 for v in self.origins.values()):
            raise ValueError("No legal windows for role/horizon")
        self.marks = None
        if dataset.startswith("ETT"):
            if dates is None:
                raise Blocked("Verified ETT timestamps missing")
            interval = np.timedelta64(15, "m") if dataset.startswith("ETTm") else np.timedelta64(1, "h")
            if not (np.diff(dates) == interval).all():
                raise ValueError("ETT time index mismatch")
            self.marks = torch.tensor(time_features(dates, dataset.startswith("ETTm")))
        self.origin_digest = identity({k: [int(v[0]), int(v[-1]), len(v)] for k, v in self.origins.items()})

    def batch(self, role, indices):
        if role not in self.origins:
            raise PermissionError("Only legal TRAIN and DEV/INNER_VAL windows are available")
        origin = torch.tensor(self.origins[role][np.asarray(indices)], dtype=torch.long)
        hx = origin[:, None] + torch.arange(-336, 0)
        hy = origin[:, None] + torch.arange(self.horizon)
        x, y = self.z[hx].to(self.device), self.z[hy].to(self.device)
        xm = None if self.marks is None else self.marks[hx].to(self.device)
        ym = None if self.marks is None else self.marks[hy].to(self.device)
        return x, y, xm, ym

    def inverse(self, values):
        return values * self.sd + self.mu

    def manifest(self):
        return dict(dataset=self.dataset, role=self.role, history=336, horizon=self.horizon, channels=self.channels,
                    cache_SHA256=self.cache_SHA256, statistics_rows=[0, self.train_end], stop=self.stop,
                    coordinate_policy=self.coordinate_policy, origin_digest=self.origin_digest,
                    TEST_access=False)


def calculate_q(raw, lag):
    if raw.ndim != 2 or raw.shape[0] <= lag or not np.isfinite(raw).all():
        raise ValueError("Invalid Q input")
    covariance, skipped = [], []
    for channel in range(raw.shape[1]):
        window = np.lib.stride_tricks.sliding_window_view(raw[:, channel], lag).T
        matrix = np.cov(window)
        diagonal = np.diag(matrix)
        if (diagonal < 1e-4).any():
            skipped.append(channel)
            continue
        covariance.append((matrix / diagonal).astype(np.float32))
    if not covariance:
        raise ValueError("No admissible covariance channels; no fallback")
    _, vectors = np.linalg.eigh(np.mean(covariance, axis=0))
    q = np.flip(vectors.T, axis=0).copy()
    if not np.allclose(q @ q.T, np.eye(lag), rtol=1e-5, atol=1e-5):
        raise ValueError("Q orthogonality failed")
    return q, skipped


def validate_q(receipt, data):
    """Check that the saved transforms match the data split and window lengths."""
    expected = data.manifest()
    for key in ("dataset", "role", "history", "horizon", "channels", "cache_SHA256", "statistics_rows"):
        if receipt.get(key) != expected[key]:
            raise ValueError("Q identity mismatch: " + key)
    if receipt.get("TEST_access") is not False:
        raise ValueError("Q access identity invalid")
    paths = {}
    for key, size in (("history", 336), ("future", data.horizon)):
        p = Path(receipt["paths"][key])
        if sha256(p) != receipt["SHA256"][key]:
            raise ValueError("Q content hash mismatch")
        q = np.load(p, allow_pickle=False)
        if q.shape != (size, size) or not np.isfinite(q).all():
            raise ValueError("Q dimension mismatch")
        paths[key] = str(p)
    return paths
