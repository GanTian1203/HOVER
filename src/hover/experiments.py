"""Run paper experiments with their required backbones and adapter selection."""
from pathlib import Path

import numpy as np
import torch

from .backbones import build, external_source
from .audit import audit_fit
from .data import Windows, calculate_q, npy_prefix, validate_q
from .io import Blocked, config, identity, load, save, sha256
from .method import HOVER, Simple
from .training import choose, fit

SEEDS = (951, 952, 953, 954, 955)
CONTROLS = ("history_only", "no_history_direct", "concatenation", "history_condition_replacement")
SYSTEMS = ("NLINEAR_STRONG", "DLINEAR", "FITS", "PATCHTST", "ITRANSFORMER", "SIGMA", "TIMEMIXER", "OLINEAR", "NLINEAR_HOVER", "OLINEAR_HOVER")


def task_matrix(stage, dataset=None, system=None, seed=None, horizon=None):
    datasets = config("datasets")
    tasks = []
    if dataset and dataset not in datasets:
        raise ValueError("Unknown dataset")
    if seed is not None and seed not in SEEDS:
        raise ValueError("Formal seed must be 951--955")
    for d in datasets:
        if dataset and dataset != d:
            continue
        if stage == "A":
            arms, lengths = SYSTEMS, (96,)
        elif stage == "B":
            arms, lengths = CONTROLS, (96,)
        elif stage == "C":
            if d not in config("horizons"):
                if dataset:
                    raise ValueError("Dataset not in the frozen C panel")
                continue
            arms, lengths = ("FROZEN", "SIMPLE", "HOVER"), config("horizons")[d]
        elif stage == "Table2":
            scope = config("paper_scope")
            if d not in scope["Table2_datasets"]:
                if dataset:
                    raise ValueError("Dataset not in frozen Table2 panel")
                continue
            arms = tuple(b + "+" + a for b in scope["Table2_backbones"] for a in scope["Table2_arms"])
            lengths = (96,)
        else:
            raise ValueError("Unknown stage")
        if system and system not in arms:
            raise ValueError("Unknown system/control for selected stage")
        if horizon is not None and horizon not in lengths:
            raise ValueError("Horizon not in frozen stage/dataset panel")
        for h in lengths:
            for arm in arms:
                for s in SEEDS:
                    if (system is None or arm == system) and (seed is None or s == seed) and (horizon is None or h == horizon):
                        tasks.append(dict(stage=stage, dataset=d, horizon=h, system=arm, seed=s, role="DEV", history=336, TEST_access=False))
    return tasks


def protocol_adapter(lr=.001, retention=1.):
    return dict(lr=lr, retention=retention, batch=64, epochs=30, minimum_epochs=5, patience=3, clip=1.,
                lradj="constant", objective="adapter", selection="best_validation", evaluation_batch=32)


class Experiment:
    def __init__(self, data_root, output, device="cpu", external_root=None):
        self.data_root, self.output = Path(data_root), Path(output)
        self.device, self.external_root = device, external_root
        package = Path(__file__).parent
        self.science_identity = identity({name: sha256(package / name) for name in (
            "method.py", "training.py", "data.py", "backbones.py", "experiments.py",
            "configuration/native.json", "configuration/inner_native.json", "configuration/paired_nlinear.json",
            "configuration/datasets.json", "configuration/sources.json", "configuration/paper_scope.json",
            "configuration/external_backbones.json")})

    def data(self, d, h, role):
        path = self.data_root / (d + ".npz")
        if not path.exists() or not path.with_suffix(".json").exists():
            raise Blocked("Prepared legal prefix missing for " + d)
        return Windows(d, path, h, role, self.device)

    def q(self, data):
        directory = self.output / "Q" / data.dataset / str(data.horizon) / data.role
        receipt_path = directory / "receipt.json"
        if receipt_path.exists():
            receipt = load(receipt_path)
            validate_q(receipt, data)
            return receipt
        # Use only the training prefix for this role. Inner Q prefers raw values
        # when available; formal Q recovers values from the stored affine scaling.
        import zipfile
        with zipfile.ZipFile(self.data_root / (data.dataset + ".npz")) as archive:
            key = "raw" if data.role == "inner" and "raw.npy" in archive.namelist() else "z"
            values = npy_prefix(archive, key, data.train_end, data.channels)
            if key == "z":
                with archive.open("mu.npy") as stream:
                    center = np.load(stream, allow_pickle=False)
                with archive.open("sd.npy") as stream:
                    scale = np.load(stream, allow_pickle=False)
                values = (values.astype(np.float64) * scale + center).astype(np.float32)
            else:
                values = values.astype(np.float32)
        directory.mkdir(parents=True, exist_ok=False)
        paths, hashes, skipped = {}, {}, {}
        for name, length in (("history", 336), ("future", data.horizon)):
            matrix, omitted = calculate_q(values, length)
            p = directory / (name + ".npy")
            np.save(p, matrix)
            paths[name], hashes[name], skipped[name] = str(p.resolve()), sha256(p), omitted
        receipt = dict(data.manifest(), paths=paths, SHA256=hashes, skipped_channels=skipped)
        save(receipt_path, receipt)
        validate_q(receipt, data)
        return receipt

    def native_recipe(self, backbone, d, seed, role):
        if backbone == "NLINEAR_PAIRED" and role == "formal":
            return None, [config("paired_nlinear")[d]["base"]]
        source_method = "NLINEAR_STRONG" if backbone == "NLINEAR_PAIRED" else backbone
        row = config("native")[source_method][d][str(seed if role == "formal" else 951)]
        args = dict(row["arguments"])
        if role == "formal":
            return args, row["candidates"]
        if backbone in ("DLINEAR", "NLINEAR_PAIRED", "NLINEAR_STRONG"):
            p = dict(lr=.001, batch=64, epochs=20, lradj="constant", objective="MSE")
        elif backbone == "FITS":
            p = dict(lr=.0005, batch=64, epochs=50, lradj="type3", objective="MSE", nested_fits=True)
        else:
            # The INNER_FIT backbone has its own training schedule.
            original = config("inner_native")[backbone][d]
            args = dict(original["arguments"])
            p = dict(lr=float(args["learning_rate"]), batch=int(args["batch_size"]), epochs=int(args["train_epochs"]),
                     lradj=args["lradj"], objective="weighted_L1" if backbone == "OLINEAR" else "MSE",
                     lossfun_alpha=float(args.get("lossfun_alpha", .5)), pct_start=float(args.get("pct_start", .3)))
        p.update(selection="fixed_last", minimum_epochs=p["epochs"], patience=None)
        return args, [p]

    def native(self, backbone, d, h, seed, role="formal", stage="A"):
        if backbone in ("PATCHTST", "ITRANSFORMER", "SIGMA", "TIMEMIXER", "OLINEAR"):
            external_source(backbone, self.external_root)
        data = self.data(d, h, role)
        q = self.q(data) if backbone == "OLINEAR" else None
        args, protocols = self.native_recipe(backbone, d, seed, role)
        if stage == "C" and h != 96 and backbone == "OLINEAR" and role == "formal":
            original = config("inner_native")["OLINEAR"][d]["arguments"]
            args = dict(original)
            protocols = [dict(lr=float(args["learning_rate"]), batch=int(args["batch_size"]), epochs=int(args["train_epochs"]),
                minimum_epochs=1, patience=int(args["patience"]), lradj=args["lradj"], objective="weighted_L1",
                lossfun_alpha=float(args["lossfun_alpha"]), selection="best_validation", evaluation_batch=16)]
        root = self.output / "native" / backbone / d / str(h) / role / str(seed)
        candidates = []
        models = []
        for number, protocol in enumerate(protocols):
            protocol = dict(protocol)
            protocol.setdefault("evaluation_batch", 32 if backbone in ("FITS", "NLINEAR_STRONG", "DLINEAR", "NLINEAR_PAIRED") else 16)
            model = build(backbone, d, seed, h, self.device, self.external_root, q, data, args)
            q_identity = None if q is None else {k: q[k] for k in ("dataset", "role", "history", "horizon", "statistics_rows", "SHA256")}
            binding = dict(backbone=backbone, dataset=d, horizon=h, role=role, seed=seed, Q=q_identity,
                           source_commit=self.source_commit(backbone), science_identity=self.science_identity,
                           arguments_identity=identity(args))
            row = fit(model, data, protocol, seed, root / ("candidate" + str(number + 1)), binding, backbone, self.external_root)
            audit_fit(root / ("candidate" + str(number + 1)))
            row["candidate_number"] = number + 1
            candidates.append(row)
            models.append(model)
        selected = candidates[0] if role == "inner" or len(candidates) == 1 else choose(candidates)
        index = selected["candidate_number"] - 1
        checkpoint = root / ("candidate" + str(index + 1)) / "best.pt"
        if sha256(checkpoint) != selected["checkpoint_SHA256"]:
            raise ValueError("Frozen native checkpoint changed")
        cp = torch.load(checkpoint, map_location="cpu")
        models[index].load_state_dict(cp["state"], strict=True)
        base_identity = dict(backbone=backbone, dataset=d, horizon=h, seed=seed, role=role,
                             checkpoint_SHA256=selected["checkpoint_SHA256"], fit_id=selected["fit_id"], Q=binding["Q"],
                             receipt_directory=(root / ("candidate" + str(index + 1))).relative_to(self.output).as_posix())
        return models[index], selected, base_identity

    @staticmethod
    def source_commit(backbone):
        source = "LTSF" if backbone in ("NLINEAR_PAIRED", "NLINEAR_STRONG", "DLINEAR") else backbone
        return config("sources")[source]["commit"]

    def selection(self, backbone, d, h, arm, stage):
        directory = self.output / "selection" / backbone / d / str(h) / arm
        frozen = directory / "frozen.json"
        inner = self.data(d, h, "inner")
        identity_value = dict(backbone=backbone, dataset=d, horizon=h, arm=arm, data=inner.manifest(),
                              development_seed=731, confirmation_seed=732,
                              science_identity=self.science_identity, TEST_access=False)
        if frozen.exists():
            result = load(frozen)
            if result["identity"] != identity_value:
                raise ValueError("Selection identity mismatch")
            for dependency in result["dependencies"]:
                if sha256(directory / dependency["receipt"]) != dependency["SHA256"]:
                    raise ValueError("Frozen candidate/confirmation receipt changed")
            return result["protocol"], identity(result)
        base731, _, ident731 = self.native(backbone, d, h, 731, "inner", stage)
        model_class = HOVER if arm == "HOVER" else Simple
        candidates, dependencies = [], []
        def candidate(protocol, seed, base, base_id, name):
            model = model_class(base, seed, horizon=h).to(self.device)
            row = fit(model, inner, protocol, seed, directory / name,
                dict(identity_value, selection_stage=name, source_commit=self.source_commit(backbone)),
                backbone, self.external_root, base_id)
            audit_fit(directory / name, base_directory=self.output / base_id["receipt_directory"])
            dependencies.append(dict(receipt=name + "/receipt.json", SHA256=sha256(directory / name / "receipt.json")))
            return row
        for i, strength in enumerate((1., 3., 1 / 3), 1):
            candidates.append(candidate(protocol_adapter(.001, strength), 731, base731, ident731, "C" + str(i)))
        selected_strength = choose(candidates)["protocol"]["retention"]
        candidates.append(candidate(protocol_adapter(.0003, selected_strength), 731, base731, ident731, "C4"))
        best = choose(candidates)
        base732, _, ident732 = self.native(backbone, d, h, 732, "inner", stage)
        confirmation = candidate(best["protocol"], 732, base732, ident732, "confirm732")
        if confirmation["protocol"] != best["protocol"] or confirmation["validation_role"] != "INNER_VAL":
            raise ValueError("Confirmation may not reselect")
        result = dict(identity=identity_value, protocol=best["protocol"], selected_fit_id=best["fit_id"],
                      confirmation_fit_id=confirmation["fit_id"], dependencies=dependencies, TEST_access=False)
        save(frozen, result)
        return best["protocol"], identity(result)

    def run(self, task):
        stage, d, h, system, seed = (task[k] for k in ("stage", "dataset", "horizon", "system", "seed"))
        if task not in task_matrix(stage, d, system, seed, h):
            raise ValueError("Task outside frozen matrix")
        variant = "full"
        full = None
        if stage == "B":
            full = self.run(task_matrix("A", d, "NLINEAR_HOVER", seed, 96)[0])
            backbone, arm, variant = "NLINEAR_PAIRED", "HOVER", system
        elif stage == "C":
            backbone, arm = "OLINEAR", system
        elif stage == "Table2":
            backbone, arm = system.split("+")
        elif system.endswith("_HOVER"):
            backbone = "NLINEAR_PAIRED" if system == "NLINEAR_HOVER" else "OLINEAR"
            arm = "HOVER"
        else:
            backbone, arm = system, "FROZEN"
        base, record, base_id = self.native(backbone, d, h, seed, stage=stage)
        if arm != "FROZEN":
            data = self.data(d, h, "formal")
            if backbone == "NLINEAR_PAIRED" and arm == "HOVER":
                protocol = config("paired_nlinear")[d]["adapter"]
                selection_id = identity(protocol)
            else:
                protocol, selection_id = self.selection(backbone, d, h, arm, stage)
            adapter = HOVER(base, seed, variant, h) if arm == "HOVER" else Simple(base, seed, h)
            record = fit(adapter.to(self.device), data, protocol, seed,
                self.output / "adapters" / backbone / d / str(h) / arm / variant / str(seed),
                dict(backbone=backbone, dataset=d, horizon=h, arm=arm, variant=variant,
                     selection_id=selection_id, source_commit=self.source_commit(backbone), science_identity=self.science_identity),
                backbone, self.external_root, base_id)
            audit_fit(self.output / "adapters" / backbone / d / str(h) / arm / variant / str(seed),
                      base_directory=self.output / base_id["receipt_directory"])
        result = dict(task, status="DONE", metrics=record["metrics"], checkpoint_SHA256=record["checkpoint_SHA256"],
            base_checkpoint_SHA256=base_id["checkpoint_SHA256"], fit_id=record["fit_id"],
            best_epoch=record["best_epoch"], epochs_run=record["epochs_run"],
            wall_seconds=record["wall_seconds"], total_parameters=record["total_parameters"],
            trainable_parameters=record["trainable_parameters"], data=record["data"], source_version="original_formal",
            independent_audit="NUMERIC_AGGREGATE_AND_CHECKPOINT_PASS")
        if full is not None:
            if full["base_checkpoint_SHA256"] != result["base_checkpoint_SHA256"]:
                raise ValueError("Full HOVER/control frozen base mismatch")
            result["paired_full"] = full
        save(self.output / "records" / stage / d / str(h) / system / (str(seed) + ".json"), result)
        return result
