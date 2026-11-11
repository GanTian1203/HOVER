"""Evaluate one preselected HOVER checkpoint pair on its held-out split."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from hover.backbones import build
from hover.io import sha256
from hover.method import HOVER

ROOT = Path(__file__).resolve().parent


def require_sha(path, expected):
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"File identity mismatch: {path.name}")


def load_series(raw_path, train_cache, definition):
    require_sha(raw_path, definition["raw_sha256"])
    require_sha(train_cache, definition["train_cache_sha256"])
    start, end = definition["test_rows"]
    channels = definition["channels"]
    if definition["format"] == "gzip_csv":
        with gzip.open(raw_path, "rb") as stream:
            raw = np.stack([np.fromstring(line.decode("ascii"), sep=",") for line in stream])
    else:
        with np.load(raw_path, allow_pickle=False) as archive:
            raw = archive["data"][:, :, 0].astype(np.float64)
    if raw.shape != (end, channels) or not np.isfinite(raw).all():
        raise ValueError("Unexpected dataset shape or nonfinite values")
    with np.load(train_cache, allow_pickle=False) as cached:
        mu, sd, z_prefix = cached["mu"], cached["sd"], cached["z"]
        digest = hashlib.sha256(mu.tobytes() + sd.tobytes()).hexdigest()
        if digest != definition["scaler_sha256"]:
            raise ValueError("TRAIN scaler identity mismatch")
        z = ((raw - mu) / sd).astype(np.float32)
        if not np.array_equal(z[:start], z_prefix) or not np.isfinite(z).all():
            raise ValueError("The standardized prefix or value mask changed")
    return torch.from_numpy(z)


def load_pair(args, plan, dataset, pair):
    require_sha(args.base, pair["base_sha256"])
    require_sha(args.adapter, pair["hover_sha256"])
    base_doc = torch.load(args.base, map_location="cpu", weights_only=True)
    adapter_doc = torch.load(args.adapter, map_location="cpu", weights_only=True)
    if base_doc.get("seed") != args.seed or adapter_doc.get("seed") != args.seed:
        raise ValueError("Checkpoint seed mismatch")
    if args.backbone == "OLINEAR":
        if args.q_history is None or args.q_future is None:
            raise ValueError("OLinear requires the two frozen full-TRAIN Q matrices")
        identity = dict(dataset=args.dataset, role="formal", history=plan["L"], horizon=plan["H"],
                        channels=dataset["channels"], cache_SHA256=dataset["train_cache_sha256"],
                        statistics_rows=dataset["train_rows"])
        q = dict(**identity, TEST_access=False, SHA256=dataset["q_sha256"],
                 paths=dict(history=str(args.q_history.resolve()), future=str(args.q_future.resolve())))
    else:
        if args.q_history is not None or args.q_future is not None:
            raise ValueError("NLinear does not use Q matrices")
        identity = dict(channels=dataset["channels"], horizon=plan["H"])
        q = None
    data = SimpleNamespace(channels=dataset["channels"], horizon=plan["H"], manifest=lambda: identity)
    base = build(args.backbone, args.dataset, args.seed, device=args.device,
                 external_root=args.external_root, q_receipt=q, data=data)
    if args.backbone == "NLINEAR_PAIRED":
        base.network.load_state_dict(base_doc["state"], strict=True)
        state = {("base.network.network." + name[5:]) if name.startswith("base.") else name: value
                 for name, value in adapter_doc["state"].items()}
    else:
        base.load_state_dict(base_doc["state"], strict=True)
        state = adapter_doc["state"]
    adapted = HOVER(base, args.seed, horizon=plan["H"]).to(args.device)
    adapted.load_state_dict(state, strict=True)
    frozen = base.state_dict()
    embedded = adapted.base.network.state_dict()
    if frozen.keys() != embedded.keys() or any(not torch.equal(t, embedded[k]) for k,t in frozen.items()):
        raise ValueError("The two systems do not share an identical frozen backbone")
    return adapted.eval().requires_grad_(False)


def score(model, z, start, end, history, horizon, batch, device):
    if not history <= start < end <= len(z) or end-start < horizon or batch < 1:
        raise ValueError("Invalid held-out window bounds")
    sse, sae = [0., 0.], [0., 0.]
    count = 0
    with torch.inference_mode():
        origins = torch.arange(start, end - horizon + 1)
        for chunk in origins.split(batch):
            x = z[chunk[:, None] + torch.arange(-history, 0)].to(device)
            y = z[chunk[:, None] + torch.arange(horizon)].to(device).double()
            hover, frozen, _ = model.components(x)
            for i, pred in enumerate((frozen, hover)):
                if pred.shape != y.shape:
                    raise ValueError("Prediction shape mismatch")
                err = pred.double() - y
                if not torch.isfinite(err).all():
                    raise ValueError("Nonfinite prediction or target")
                sse[i] += err.square().sum((1, 2)).sum().item()
                sae[i] += err.abs().sum((1, 2)).sum().item()
            count += y.numel()
    return [dict(system=arm, MSE=ss/count, MAE=sa/count, SSE=ss, SAE=sa, valid_count=count)
            for arm, ss, sa in zip(("Frozen", "HOVER"), sse, sae)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("ECL","Solar","PEMS03","Exchange"), required=True)
    parser.add_argument("--backbone", choices=("OLINEAR","NLINEAR_PAIRED"), required=True)
    parser.add_argument("--seed", type=int, choices=range(951, 956), required=True)
    for name in ("raw","train-cache","base","adapter","external-root"):
        parser.add_argument("--"+name, type=Path, required=True)
    for name in ("q-history","q-future","output"):
        parser.add_argument("--"+name, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    plan_path = ROOT / "test_protocol.json"
    plan = json.loads(plan_path.read_text(encoding="utf8"))
    definition = plan["datasets"][args.dataset]
    key = f"{args.dataset}/{args.backbone}/{args.seed}"
    pair = plan["pairs"][key]
    model = load_pair(args, plan, definition, pair)
    z = load_series(args.raw, args.train_cache, definition)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    result = dict(role="TEST", dataset=args.dataset, backbone=args.backbone, seed=args.seed,
                  source_sha256=definition["raw_sha256"], test_rows=definition["test_rows"],
                  protocol_sha256=sha256(plan_path), **pair,
                  results=score(model, z, *definition["test_rows"], plan["L"], plan["H"], plan["batch"], args.device))
    text = json.dumps(result, indent=2) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        with args.output.open("x", encoding="utf8") as stream:
            stream.write(text)


if __name__ == "__main__":
    main()

