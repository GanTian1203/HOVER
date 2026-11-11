"""Train and evaluate forecasters with resumable epoch checkpoints."""
import hashlib
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from .backbones import learning_rate_step
from .io import identity, load, save, sha256
from .method import adapter_loss


def improved(value, best):
    return math.isfinite(value) and (not math.isfinite(best) or value < best - max(1e-12, 1e-8 * abs(best)))


def choose(records):
    best = None
    for record in records:
        value = float(record["metrics"]["mse"])
        if not math.isfinite(value):
            raise ValueError("Nonfinite candidate MSE")
        if best is None or improved(value, float(best["metrics"]["mse"])):
            best = record
    if best is None:
        raise ValueError("No candidates")
    return best


def state_digest(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def atomic_torch(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp-" + str(os.getpid()))
    torch.save(value, temporary)
    os.replace(temporary, path)


def evaluate(model, data, batch=32, aggregate_path=None):
    model.eval()
    sse, sae, counts, origins = [], [], [], []
    channel_sse, horizon_sse = np.zeros(data.channels), np.zeros(data.horizon)
    with torch.no_grad():
        for start in range(0, len(data.origins["DEV"]), batch):
            ii = np.arange(start, min(start + batch, len(data.origins["DEV"])))
            x, y, xm, ym = data.batch("DEV", ii)
            prediction = model(x, xm, ym)
            if prediction.shape != y.shape or not torch.isfinite(prediction).all():
                raise ValueError("Invalid evaluation output")
            error = prediction.double() - y.double()
            sq = error.square()
            sse.extend(sq.sum((1, 2)).cpu().tolist())
            sae.extend(error.abs().sum((1, 2)).cpu().tolist())
            counts.extend([data.horizon * data.channels] * len(ii))
            origins.extend(data.origins["DEV"][ii].tolist())
            channel_sse += sq.sum((0, 1)).cpu().numpy()
            horizon_sse += sq.sum((0, 2)).cpu().numpy()
    if not counts or sum(counts) <= 0:
        raise ValueError("No valid evaluation elements")
    result = dict(mse=sum(sse) / sum(counts), mae=sum(sae) / sum(counts), SSE=sum(sse), SAE=sum(sae), valid_count=sum(counts))
    if aggregate_path is not None:
        np.savez_compressed(aggregate_path, origins=origins, SSE=sse, SAE=sae, valid_count=counts,
                            channel_SSE=channel_sse, horizon_SSE=horizon_sse)
    return result


def fit(model, data, protocol, seed, directory, binding, method, external_root=None, base_identity=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / "fit.lock"
    # OS locks are released after a crash. Keep the file in place so concurrent
    # callers cannot acquire locks on two different replacement files.
    stream = lock.open("a+b")
    stream.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        stream.close()
        raise RuntimeError("Fit already owned by another process") from error
    try:
        return _fit(model, data, protocol, seed, directory, binding, method, external_root, base_identity)
    finally:
        stream.seek(0)
        if os.name == "nt":
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def _fit(model, data, protocol, seed, directory, binding, method, external_root=None, base_identity=None):
    """Resume a matching run at an epoch boundary or return its completed result."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    definition = dict(data=data.manifest(), protocol=protocol, seed=seed, binding=binding,
                      base_identity=base_identity, method=method)
    fit_id = identity(definition)
    receipt_path = directory / "receipt.json"
    checkpoint = directory / "best.pt"
    resume_path = directory / "resume.pt"
    if receipt_path.exists():
        previous = load(receipt_path)
        if previous["fit_id"] != fit_id or previous["status"] != "DONE" or sha256(checkpoint) != previous["checkpoint_SHA256"]:
            raise ValueError("Completed fit identity mismatch")
        return previous
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("No trainable parameters")
    optimizer = torch.optim.Adam(parameters, lr=protocol["lr"])
    generator = np.random.default_rng(seed)
    frozen = state_digest(model.base) if hasattr(model, "base") else None
    best, best_epoch, best_state, bad, first, updates, trace = float("inf"), None, None, 0, 1, 0, []
    scheduler = None
    args = getattr(model, "arguments", {})
    if protocol["lradj"] == "TST":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=protocol["lr"],
            steps_per_epoch=math.ceil(len(data.origins["TRAIN"]) / protocol["batch"]),
            epochs=protocol["epochs"], pct_start=protocol.get("pct_start", .3))
    elapsed_before = 0.
    limit = protocol.get("initial_epoch_cap", protocol["epochs"])
    if resume_path.exists():
        saved = torch.load(resume_path, map_location="cpu")
        if saved["fit_id"] != fit_id:
            raise ValueError("Resume data/model/seed/role/Q identity mismatch")
        model.load_state_dict(saved["state"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        generator.bit_generator.state = saved["numpy_rng"]
        torch.set_rng_state(saved["torch_rng"])
        if str(data.device).startswith("cuda"):
            torch.cuda.set_rng_state(saved["cuda_rng"])
        if scheduler:
            scheduler.load_state_dict(saved["scheduler"])
        first, best, best_epoch, best_state, bad = saved["epoch"] + 1, saved["best"], saved["best_epoch"], saved["best_state"], saved["bad"]
        updates, trace, limit, elapsed_before = saved["updates"], saved["trace"], saved["limit"], saved["elapsed_seconds"]
        if frozen is not None and state_digest(model.base) != frozen:
            raise ValueError("Resumed frozen backbone changed")
    fixed = protocol["selection"] == "fixed_last"
    adapter = protocol["objective"] == "adapter"
    if adapter and base_identity is None:
        raise ValueError("Adapter base identity missing")
    started = time.monotonic()
    epoch = first - 1
    stopped = not fixed and epoch >= protocol["minimum_epochs"] and bad >= protocol["patience"]
    while epoch < limit and not stopped:
        epoch += 1
        model.train()
        order = generator.permutation(len(data.origins["TRAIN"]))
        total, count = 0., 0
        for start in range(0, len(order), protocol["batch"]):
            x, y, xm, ym = data.batch("TRAIN", order[start:start + protocol["batch"]])
            optimizer.zero_grad()
            if adapter:
                prediction, baseline, _ = model.components(x, xm, ym)
                loss, _, _ = adapter_loss(prediction, y, baseline, protocol["retention"])
            elif protocol["objective"] == "reconstruct_plus_future":
                loss = (model.reconstruction(x) - torch.cat([x, y], 1)).square().mean()
            else:
                prediction = model(x, xm, ym)
                if protocol["objective"] == "weighted_L1":
                    weights = torch.arange(1, data.horizon + 1, device=y.device, dtype=y.dtype).pow(-protocol["lossfun_alpha"])[None, :, None]
                    loss = ((prediction - y).abs() * weights).mean()
                else:
                    loss = (prediction - y).square().mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training objective")
            loss.backward()
            if adapter:
                if any(p.grad is not None for p in model.base.parameters()):
                    raise ValueError("Frozen backbone received gradients")
                torch.nn.utils.clip_grad_norm_(parameters, protocol["clip"])
            optimizer.step()
            updates += 1
            total += float(loss) * len(x)
            count += len(x)
            if scheduler:
                learning_rate_step(protocol, optimizer, epoch, args, method, external_root, scheduler)
                scheduler.step()
        validation = None if fixed else evaluate(model, data, protocol.get("evaluation_batch", 32))
        trace.append(dict(epoch=epoch, TRAIN_objective=total / count, validation=validation,
                          updates=updates, lr=optimizer.param_groups[0]["lr"]))
        value = total / count if fixed else validation["mse"]
        if fixed or improved(value, best):
            best, best_epoch, bad = value, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if protocol.get("FITS_extension") and epoch == 50:
            last = [r["validation"]["mse"] for r in trace[-3:]]
            if best_epoch == 50 and last[0] > last[1] > last[2]:
                limit = 100
        stopped = not fixed and epoch >= protocol["minimum_epochs"] and bad >= protocol["patience"]
        if not scheduler:
            learning_rate_step(protocol, optimizer, epoch, args, method, external_root)
        elapsed = elapsed_before + time.monotonic() - started
        atomic_torch(resume_path, dict(fit_id=fit_id, epoch=epoch, state=model.state_dict(),
            optimizer=optimizer.state_dict(), numpy_rng=generator.bit_generator.state, torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state() if str(data.device).startswith("cuda") else None,
            scheduler=scheduler.state_dict() if scheduler else None, best=best, best_epoch=best_epoch,
            best_state=best_state, bad=bad, updates=updates, trace=trace, limit=limit, elapsed_seconds=elapsed))
        save(directory / "curve.json", trace)
    if best_state is None:
        raise ValueError("No completed model state")
    model.load_state_dict(best_state, strict=True)
    if frozen is not None and state_digest(model.base) != frozen:
        raise ValueError("Frozen backbone changed")
    metrics = None if data.role == "inner" and fixed else evaluate(model, data, protocol.get("evaluation_batch", 32), directory / "DEV_or_INNER_VAL_aggregates.npz")
    if not fixed and abs(metrics["mse"] - best) > 1e-10:
        raise ValueError("Best checkpoint metric reload mismatch")
    atomic_torch(checkpoint, dict(fit_id=fit_id, definition=definition, state=best_state, best_epoch=best_epoch))
    receipt = dict(status="DONE", fit_id=fit_id, **definition, metrics=metrics, checkpoint_SHA256=sha256(checkpoint),
        best_epoch=best_epoch, epochs_run=epoch, updates=updates,
        validation_role="INNER_VAL" if data.role == "inner" else "DEV",
        source_version="original_formal", frozen_base_unchanged=frozen is not None,
        wall_seconds=elapsed_before + time.monotonic() - started,
        trainable_parameters=sum(p.numel() for p in parameters), total_parameters=sum(p.numel() for p in model.parameters()),
        TEST_access=False)
    save(receipt_path, receipt)
    return receipt
