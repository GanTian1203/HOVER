"""Rebuild the published DEV figures from pinned observations and forecasts."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from hover.io import config, sha256
from hover.prepare import prepare

HERE = Path(__file__).resolve().parent
DATASETS = ("ECL", "PEMS03")
ARMS6 = {"Frozen": "y_base", "HOVER": "y_hover"}
ARMS7 = {arm: arm for arm in ("Frozen", "Simple", "PostY", "HOVER")}
HISTORY, HORIZON = 336, 96


def read_csv(name):
    with (HERE / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(name):
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def window(values, definition, channel, origin):
    start, stop = definition["DEV_rows"]
    if not (0 <= channel < definition["channels"] and
            max(HISTORY, start) <= origin and origin + HORIZON <= stop):
        raise ValueError("The requested case is outside the admitted DEV windows")
    return values[origin-HISTORY:origin, channel], values[origin:origin+HORIZON, channel]


def predictions(rows, step, arms):
    ordered = sorted(rows, key=lambda row: int(row[step]))
    if [int(row[step]) for row in ordered] != list(range(HORIZON)):
        raise ValueError("Missing or duplicate forecast step")
    values = {arm: np.array([row[column] for row in ordered], dtype=np.float32)
              for arm, column in arms.items()}
    if not all(np.isfinite(series).all() for series in values.values()):
        raise ValueError("Non-finite forecast")
    return values


def scores(dataset, kind, target, arms, segments, references):
    result = []
    for segment, start, stop in segments:
        for arm, forecast in arms.items():
            error = forecast[start:stop].astype(np.float64) - target[start:stop]
            mse, mae = float(np.mean(error**2)), float(np.mean(np.abs(error)))
            reference = references[dataset, kind, segment, arm]
            np.testing.assert_allclose(
                [mse, mae], [float(reference["MSE"]), float(reference["MAE"])],
                atol=1e-7, rtol=1e-6,
                err_msg=f"{dataset}/{kind}/{segment}/{arm}",
            )
            result.append(dict(dataset=dataset, case=kind, seed=951, role="DEV",
                               segment=segment, start=start, stop=stop, arm=arm, MSE=mse, MAE=mae))
    return result


def rebuild(ecl, pems03, output, checkpoint_spec=None):
    import draw_figure6
    import draw_figure7
    from single_style import export

    if checkpoint_spec:
        from replay_checkpoints import isolated_replay, compare_replay
    definitions = config("datasets")
    output.mkdir(parents=True, exist_ok=False)
    values = {}
    for dataset, source in zip(DATASETS, (ecl, pems03)):
        cache = output / f"{dataset}.npz"
        meta = prepare(dataset, source, cache)
        with np.load(cache, allow_pickle=False) as archive:
            values[dataset] = archive["z"]
        expected = read_json("waveform_identity.json")["standardized_prefix_sha256"][dataset]
        if meta["canonical_z_SHA256"] != expected:
            raise ValueError(f"{dataset}: standardized TRAIN/DEV prefix identity mismatch")

    cases6 = read_json("figure6_source_manifest.json")["results"]
    rows6 = read_csv("figure6_selected_forecasts.csv")
    refs6 = {(r["dataset"], "selected", 0, r["system"]): r
             for r in read_csv("figure6_window_scores.csv")}
    traces6, checks = [], []
    for case in cases6:
        dataset = case["dataset"]
        origin, channel, seed = (case[k] for k in ("origin", "channel_0_based", "seed"))
        if case["role"] != "DEV" or case["horizon"] != HORIZON or case["history"] != HISTORY or seed != 951:
            raise ValueError("Unexpected Figure 6 case identity")
        history, truth = window(values[dataset], definitions[dataset], channel, origin)
        rows = [r for r in rows6 if (r["dataset"], int(r["seed"]), int(r["origin"]),
                                    int(r["channel_0_based"])) == (dataset, seed, origin, channel)]
        arms = predictions(rows, "relative_step", ARMS6)
        if checkpoint_spec:
            arms = compare_replay(isolated_replay(checkpoint_spec, 6, case, output / f"{dataset}.npz"), arms)
        checks.extend(scores(dataset, "selected", truth, arms, [(0, 0, HORIZON)], refs6))
        for step in range(-HISTORY, HORIZON):
            row = dict(dataset=dataset, seed=seed, origin=origin, channel_0_based=channel,
                       relative_step=step)
            if step < 0:
                row["history"] = history[step]
            else:
                row.update(y_true=truth[step], y_base=arms["Frozen"][step], y_hover=arms["HOVER"][step])
            traces6.append(row)

    cases7 = read_json("selected_cases.json")["cases"]
    rows7 = read_csv("figure7_selected_forecasts.csv")
    seed_rows = read_csv("FIG7_SEED_CONSISTENCY.csv")
    refs7 = {(r["dataset"], r["kind"], int(r["segment"]), r["system"]): r
             for r in seed_rows if r["seed"] == "951"}
    traces7 = {}
    segments = [(0, 0, HORIZON)] + [(i + 1, i * 24, (i + 1) * 24) for i in range(4)]
    for case in cases7:
        dataset, kind, seed = (case[k] for k in ("dataset", "kind", "seed"))
        if case["evaluation_role"] != "DEV" or case["horizon"] != HORIZON or seed != 951:
            raise ValueError("Unexpected Figure 7 case identity")
        if case["origin"] != case["prediction_start_row"]:
            raise ValueError("Figure 7 forecast start mismatch")
        history, truth = window(values[dataset], definitions[dataset], case["channel"], case["origin"])
        rows = [r for r in rows7 if (r["dataset"], r["case"], int(r["seed"])) == (dataset, kind, seed)]
        arms = predictions(rows, "forecast_step", ARMS7)
        reconstruction = predictions(rows, "forecast_step", {"HOVER": "reconstruction"})
        np.testing.assert_allclose(reconstruction["HOVER"], arms["HOVER"], atol=1e-6, rtol=1e-5)
        if checkpoint_spec:
            arms = compare_replay(isolated_replay(checkpoint_spec, 7, case, output / f"{dataset}.npz"), arms)
        for ref in (r for r in seed_rows if r["dataset"] == dataset and r["kind"] == kind and r["seed"] == str(seed)):
            if (int(ref["channel"]), int(ref["origin"]), ref["evaluation_role"]) != (
                    case["channel"], case["origin"], "DEV"):
                raise ValueError("Figure 7 score refers to a different case")
        checks.extend(scores(dataset, kind, truth, arms, segments, refs7))
        prefix = f"{dataset}_{kind}_{seed}"
        traces7.update({f"{prefix}_{name}": array for name, array in
                        dict(history=history, truth=truth, **arms).items()})

    export(draw_figure6.draw((cases6, traces6)), "figure6_en", output)
    draw_figure7.draw(cases7, seed_rows, traces7, output)
    with (output / "window_metrics.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, checks[0].keys())
        writer.writeheader()
        writer.writerows(checks)
    print(f"Verified {len(checks)} paired window/segment MSE and MAE rows. Figures: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ecl", type=Path, required=True, help="Pinned electricity.txt.gz")
    parser.add_argument("--pems03", type=Path, required=True, help="Pinned PEMS03.npz")
    parser.add_argument("--output", type=Path, required=True, help="New local output directory")
    parser.add_argument("--checkpoints", type=Path, help="Optional local specification for exact checkpoint replay")
    args = parser.parse_args()
    for dataset, source in zip(DATASETS, (args.ecl, args.pems03)):
        definition = config("datasets")[dataset]
        if not source.is_file():
            parser.error(f"{dataset}: missing {source}. Obtain {definition['source_url']}")
        if sha256(source) != definition["source_full_SHA256"]:
            parser.error(f"{dataset}: source SHA256 mismatch. Use {definition['source_url']}")
    rebuild(args.ecl, args.pems03, args.output, args.checkpoints)


if __name__ == "__main__":
    main()
