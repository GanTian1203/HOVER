"""Load the exact historical checkpoints for Figures 6 and 7."""
import ast
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import torch

from hover.backbones import build
from hover.io import sha256
from hover.method import HOVER, Simple

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def require_sha(path, expected):
    if sha256(path) != expected:
        raise ValueError(f"Incorrect file identity: {path.name}")


def state(path, expected, dataset, seed):
    require_sha(path, expected)
    document = torch.load(path, map_location="cpu", weights_only=True)
    if (document.get("dataset"), document.get("seed")) != (dataset, seed):
        raise ValueError("Checkpoint dataset or seed mismatch")
    return document


class CheckpointReplay:
    def __init__(self, specification):
        torch.set_num_threads(1)
        self.spec = read(specification)
        self.directory = specification.resolve().parent
        self.protocol = read(HERE.parent / "test/test_protocol.json")
        self.dlinear = read(HERE / "figure7_checkpoint_identity.json")
        self.f6_models = {}
        self.f7_models = {}

    def figure6(self, case, series):
        dataset, backbone, seed = (case[key] for key in ("dataset", "backbone", "seed"))
        paths = self.spec["figure6"][dataset]
        pair = self.protocol["pairs"][f"{dataset}/{backbone}/{seed}"]
        if case["checkpoint_SHA256"] != [pair["base_sha256"], pair["hover_sha256"]]:
            raise ValueError("Figure 6 and the reference checkpoint inventory disagree")
        if dataset not in self.f6_models:
            # This loader validates weights and Q. It does not open evaluation data.
            sys.path.insert(0, str(HERE.parent / "test"))
            from test_holdout import load_pair

            args = SimpleNamespace(dataset=dataset, backbone=backbone, seed=seed, device="cpu",
                                   base=self.directory / paths["base"], adapter=self.directory / paths["hover"],
                                   q_history=self.directory / paths["q_history"] if "q_history" in paths else None,
                                   q_future=self.directory / paths["q_future"] if "q_future" in paths else None,
                                   external_root=self.directory / self.spec["external_root"])
            self.f6_models[dataset] = load_pair(args, self.protocol, self.protocol["datasets"][dataset], pair)
        origin, channel = case["origin"], case["channel_0_based"]
        x = torch.from_numpy(series[None, origin-336:origin].copy())
        with torch.inference_mode():
            refined, frozen, _ = self.f6_models[dataset].components(x)
        return {"Frozen": frozen[0, :, channel].numpy(), "HOVER": refined[0, :, channel].numpy()}

    def dlinear_models(self, dataset, channels):
        seed = 951
        paths = self.spec["figure7"][dataset]
        expected = self.dlinear[dataset]
        base = build("DLINEAR", dataset, seed, external_root=self.directory / self.spec["external_root"],
                     data=SimpleNamespace(channels=channels, horizon=96))
        checkpoint = state(self.directory / paths["base"], expected["base"], dataset, seed)
        base.network.load_state_dict(checkpoint["state"], strict=True)
        base.eval().requires_grad_(False)
        simple, hover = Simple(base, seed), HOVER(base, seed)
        for name, model, embedded in (("simple", simple, simple.base), ("hover", hover, hover.base.network)):
            checkpoint = state(self.directory / paths[name], expected[name], dataset, seed)
            if checkpoint["base_identity"]["checkpoint_SHA256"] != expected["base"]:
                raise ValueError("DLinear adapter is bound to a different base checkpoint")
            model.load_state_dict(checkpoint["state"], strict=True)
            if any(not torch.equal(weight, embedded.state_dict()[key]) for key, weight in base.state_dict().items()):
                raise ValueError("Embedded DLinear weights differ from Frozen")
            model.eval().requires_grad_(False)

        posty = read(HERE.parent / "posty/protocol.json")
        source_root = self.directory / self.spec["posty_root"]
        commit = subprocess.check_output(["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip()
        if commit != posty["upstream_commit"]:
            raise ValueError("Incorrect PostY source commit")
        source = source_root / "Adapter-X+Y/experiments/exp_post_y_mul.py"
        require_sha(source, posty["upstream_model_SHA256"])
        tree = ast.parse(source.read_text(encoding="utf-8"))
        node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PostY")
        namespace = {"torch": torch, "F": torch.nn.functional}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
        post = namespace["PostY"](96 * channels, posty["config"]["hidden"], 96 * channels)
        checkpoint = state(self.directory / paths["posty"], expected["posty"], dataset, seed)
        if checkpoint["base_identity"]["checkpoint_SHA256"] != expected["base"]:
            raise ValueError("PostY is bound to a different DLinear checkpoint")
        post.load_state_dict(checkpoint["adapter_state"], strict=True)
        return base, simple, post.eval().requires_grad_(False), hover

    def figure7(self, case, series):
        dataset, origin, channel = (case[key] for key in ("dataset", "origin", "channel"))
        if dataset not in self.f7_models:
            self.f7_models[dataset] = self.dlinear_models(dataset, series.shape[1])
        base, simple, post, hover = self.f7_models[dataset]
        x = torch.from_numpy(series[None, origin-336:origin].copy())
        with torch.inference_mode():
            frozen = base(x)
            result = {"Frozen": frozen, "Simple": simple(x),
                      "PostY": frozen * post(frozen.flatten(1)).view_as(frozen), "HOVER": hover(x)}
        return {name: pred[0, :, channel].numpy() for name, pred in result.items()}


def compare_replay(recomputed, published):
    for name, forecast in recomputed.items():
        np.testing.assert_allclose(forecast, published[name], atol=2e-6, rtol=1e-5, err_msg=name)
    return recomputed


def isolated_replay(specification, figure, case, cache):
    """Run the pinned upstream import in a fresh Python process."""
    import tempfile

    with tempfile.TemporaryDirectory() as temp:
        result = Path(temp) / "prediction.json"
        subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()),
                        str(specification.resolve()), str(figure), json.dumps(case),
                        str(cache.resolve()), str(result)], check=True)
        return {name: np.asarray(values, dtype=np.float32) for name, values in read(result).items()}


if __name__ == "__main__":
    spec, figure, case, cache, result = sys.argv[1:]
    with np.load(cache, allow_pickle=False) as archive:
        series = archive["z"]
    replay = CheckpointReplay(Path(spec))
    function = { "6": replay.figure6, "7": replay.figure7 }[figure]
    result_values = function(json.loads(case), series)
    Path(result).write_text(json.dumps({key: value.tolist() for key, value in result_values.items()}),
                            encoding="utf-8")
