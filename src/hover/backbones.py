"""Load pinned upstream forecasters behind a common window-to-forecast interface."""
import importlib.util
import inspect
from pathlib import Path
import sys
import os
from types import SimpleNamespace

import torch
from torch import nn

from .data import validate_q
from .io import Blocked, config, load, sha256


def source_key(method):
    return "LTSF" if method.upper() in ("NLINEAR", "NLINEAR_PAIRED", "NLINEAR_STRONG", "DLINEAR") else method.upper()


def external_source(method, external_root):
    """Resolve an upstream source tree and verify its pinned files before import."""
    key = source_key(method)
    source = config("sources")[key]
    root = Path(external_root or os.environ.get("HOVER_EXTERNAL_ROOT", "external_models")) / key
    if not root.is_dir():
        raise Blocked("Exact external source missing for " + method + "; use hover sources or a separately obtained source directory")
    acknowledgement = root / ("USER_SUPPLIED_SOURCE.json" if key == "OLINEAR" else "ACQUISITION.json")
    if acknowledgement.exists() or key == "OLINEAR":
        admitted = load(acknowledgement) if acknowledgement.exists() else {}
        if admitted.get("commit") != source["commit"] or (key == "OLINEAR" and not admitted.get("permission_basis")):
            raise Blocked("External source commit/permission-basis identity missing or mismatched: " + key)
        # A local registration can point to a separate source tree.
        if admitted.get("source_directory"):
            candidate = Path(admitted["source_directory"])
            root = (candidate if candidate.is_absolute() else root / candidate).resolve()
            if not root.is_dir():
                raise Blocked("User-supplied OLinear source directory is missing")
    for name, expected in source["files"].items():
        if name.endswith(".py") or name in source["license_files"]:
            path = root / name
            if not path.is_file() or sha256(path) != expected:
                raise Blocked("External source identity mismatch: " + method + "/" + name)
    return root


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Native(nn.Module):
    """Adapt upstream model calls to [batch, horizon, channels] forecasts."""

    def __init__(self, network, method, arguments):
        super().__init__()
        self.network, self.method, self.arguments = network, method, arguments

    def reconstruction(self, x):
        if self.method != "FITS":
            raise ValueError("Reconstruction objective belongs to FITS")
        return self.network(x)[0]

    def forward(self, x, xmark=None, future_mark=None):
        a = self.arguments
        horizon = a["pred_len"]
        if x.ndim != 3 or x.shape[1:] != (336, a["enc_in"]):
            raise ValueError("Native history/channel identity mismatch")
        if a.get("freq") == "h":
            if xmark is not None and xmark.shape[-1] == 5:
                xmark = xmark[:, :, 1:]
            if future_mark is not None and future_mark.shape[-1] == 5:
                future_mark = future_mark[:, :, 1:]
        if self.method in ("NLINEAR_PAIRED", "NLINEAR_STRONG", "DLINEAR", "PATCHTST"):
            prediction = self.network(x)
        elif self.method == "FITS":
            prediction = self.network(x)[0][:, -horizon:]
        else:
            label = int(a.get("label_len", 0))
            decoder = torch.cat([x[:, -label:] if label else x[:, :0], x.new_zeros((len(x), horizon, x.shape[2]))], 1)
            prediction = self.network(x, xmark, decoder, future_mark)
        if isinstance(prediction, tuple):
            prediction = prediction[0]
        if prediction.shape != (len(x), horizon, a["enc_in"]):
            raise ValueError("Native output horizon/channel mismatch")
        return prediction


def build(method, dataset, seed, horizon=96, device="cpu", external_root=None, q_receipt=None, data=None, arguments=None):
    if seed not in (731, 732, 951, 952, 953, 954, 955):
        raise ValueError("Native model seed outside development/formal protocol")
    profile_seed = seed if seed in range(951, 956) else 951
    if method == "NLINEAR_PAIRED":
        # Paired and Strong NLinear share source code and use separate recipes.
        args = dict(seq_len=336, pred_len=horizon, enc_in=data.channels, individual=False)
    else:
        args = dict(arguments if arguments is not None else config("native")[method][dataset][str(profile_seed)]["arguments"])
        args["pred_len"] = horizon
    if data is None or args["enc_in"] != data.channels or args["seq_len"] != 336 or data.horizon != horizon:
        raise ValueError("Native data interface mismatch")
    torch.manual_seed(seed)
    if method in ("NLINEAR_PAIRED", "NLINEAR_STRONG", "DLINEAR", "FITS"):
        root = external_source(method, external_root)
        name = "NLinear" if method.startswith("NLINEAR") else "DLinear" if method == "DLINEAR" else "FITS"
        if method in ("DLINEAR", "FITS"):
            # Upstream absolute imports can collide across backbone families.
            for package in ("layers", "utils", "models", "model"):
                imported = sys.modules.get(package)
                imported_path = getattr(imported, "__file__", None)
                if imported_path and root.resolve() not in Path(imported_path).resolve().parents:
                    raise Blocked("Use a separate process per external backbone")
            sys.path.insert(0, str(root))
        Model = import_file("hover_native_" + method, root / "models" / (name + ".py")).Model
    else:
        root = external_source(method, external_root)
        recipe = config("native")[method][dataset][str(profile_seed)]
        import_root = str(root / recipe["import_subdir"])
        for package in ("layers", "utils", "models", "model"):
            imported = sys.modules.get(package)
            imported_path = getattr(imported, "__file__", None)
            if imported_path and root.resolve() not in Path(imported_path).resolve().parents:
                raise Blocked("Use a separate process for each external backbone to avoid module namespace collisions")
        sys.path.insert(0, import_root)
        if method == "OLINEAR":
            if q_receipt is None:
                raise Blocked("Role-specific OLinear Q receipt missing")
            paths = validate_q(q_receipt, data)
            args.update(q_mat_file=paths["history"], q_out_mat_file=paths["future"], q_channel_file="", root_path=str(Path(paths["history"]).parent))
            if args.get("Q_chan_indep") or args.get("Q_loss"):
                raise Blocked("Unadmitted OLinear Q mechanism")
        try:
            Model = import_file("hover_native_" + method, root / recipe["module_relative"]).Model
        except ImportError as error:
            raise Blocked("Full official " + method + " dependency unavailable: " + str(error)) from error
        if "data_provider.data_loader" in sys.modules:
            raise Blocked("Default official data loader imported; use the reviewed prefix interface")
    return Native(Model(SimpleNamespace(**args)), method, args).to(device)


def adjuster(method, external_root):
    root = external_source(method, external_root)
    path = root / ("PatchTST_supervised/utils/tools.py" if method == "PATCHTST" else "utils/tools.py")
    return import_file("hover_adjuster_" + method, path).adjust_learning_rate


def learning_rate_step(protocol, optimizer, epoch, args, method, external_root, scheduler=None):
    schedule = protocol["lradj"]
    if schedule == "constant":
        return
    if schedule in ("type1", "type3", "p3_type1") and method in ("NLINEAR_PAIRED", "NLINEAR_STRONG", "DLINEAR", "FITS"):
        # Nested FITS decays at epoch 1; the other recipes start at exponent zero.
        exponent = epoch if protocol.get("nested_fits") else epoch - 1
        factor = .95 if schedule == "type3" else .5
        for group in optimizer.param_groups:
            group["lr"] = protocol["lr"] * factor ** exponent
        return
    fn = adjuster(method, external_root)
    supplied = dict(optimizer=optimizer, scheduler=scheduler, epoch=epoch, args=SimpleNamespace(**args), printout=False)
    fn(**{key: supplied[key] for key in inspect.signature(fn).parameters})
