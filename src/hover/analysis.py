"""Five-seed DEV tables and same-backbone paired differences from numeric records."""
import collections
import csv
from decimal import Decimal
import math
from pathlib import Path
import statistics

from .io import load


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Cannot publish an empty table")
    with path.open("w", encoding="utf8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def records_from_csv(path, kind="main"):
    rows = list(csv.DictReader(Path(path).open(encoding="utf-8-sig", newline="")))
    result = []
    for row in rows:
        if row.get("TEST_access", "False") != "False":
            raise ValueError("Unaccepted reference access")
        systems = [(row["system"], "MSE", "MAE")] if kind == "main" else [
            (row["method"], "MSE", "MAE"), ("HOVER", "full_HOVER_MSE", "full_HOVER_MAE")]
        for system, mse, mae in systems:
            base = row.get("base_checkpoint_SHA256", "")
            result.append(dict(dataset=row["dataset"], horizon=96, seed=int(row["seed"]), system=system,
                role="DEV", status="DONE", source_version="original_formal", TEST_access=False,
                base_checkpoint_SHA256=base,
                checkpoint_SHA256=row.get("full_HOVER_SHA256", "") if system == "HOVER" else row["checkpoint_SHA256"],
                metrics=dict(mse=float(row[mse]), mae=float(row[mae]), valid_count=int(row.get("valid_count") or row["valid_targets"]))))
    unique = {}
    for row in result:
        key = (row["dataset"], row["horizon"], row["system"], row["seed"])
        if key in unique and unique[key] != row:
            raise ValueError("Conflicting duplicate reference result")
        unique[key] = row
    return list(unique.values())


def read_records(directory, stage=None):
    root = Path(directory)
    if stage:
        root = root / stage
    rows = []
    for path in root.rglob("*.json"):
        row = load(path)
        if row.get("status") == "DONE" and "system" in row and "metrics" in row:
            rows.append(row)
    if stage == "B":
        full = {}
        for row in rows:
            if "paired_full" in row:
                value = dict(row["paired_full"], system="HOVER")
                key = (value["dataset"], value["horizon"], value["seed"])
                if key in full and full[key] != value:
                    raise ValueError("Conflicting Full HOVER control reference")
                full[key] = value
        rows.extend(full.values())
    return rows


def summarize(rows):
    groups = collections.defaultdict(list)
    seen = set()
    for row in rows:
        if row["role"] != "DEV" or row["TEST_access"] is not False or row["status"] != "DONE":
            raise ValueError("Only accepted DEV records may enter a formal table")
        if row["seed"] not in range(951, 956) or row["metrics"]["valid_count"] <= 0:
            raise ValueError("Invalid formal seed or element count")
        key = (row["dataset"], row["horizon"], row["system"], row["seed"])
        if key in seen:
            raise ValueError("Duplicate model/dataset/horizon/seed")
        seen.add(key)
        groups[key[:3]].append(row)
    table = []
    for (dataset, horizon, system), group in groups.items():
        if {r["seed"] for r in group} != set(range(951, 956)):
            raise ValueError("Five formal seeds are required per group")
        if len({r["metrics"]["valid_count"] for r in group}) != 1:
            raise ValueError("Evaluation mask/origin count mismatch")
        if len({r["source_version"] for r in group}) != 1:
            raise ValueError("Cannot mix configuration versions")
        value = dict(dataset=dataset, horizon=horizon, system=system, seeds=5, role="DEV",
                     valid_count=group[0]["metrics"]["valid_count"], source_version=group[0]["source_version"])
        for metric in ("mse", "mae"):
            numbers = [r["metrics"][metric] for r in group]
            if not all(math.isfinite(x) for x in numbers):
                raise ValueError("Nonfinite result")
            value[metric + "_mean"] = statistics.mean(numbers)
            value[metric + "_SD"] = statistics.stdev(numbers)
        table.append(value)
    return table


def placements(table):
    grouped = collections.defaultdict(list)
    for row in table:
        grouped[row["dataset"], row["horizon"]].append(row)
    awards = []
    for (d, h), group in grouped.items():
        if len({r["valid_count"] for r in group}) != 1:
            raise ValueError("System-level evaluation count mismatch")
        for metric in ("mse", "mae"):
            remaining = sorted(group, key=lambda r: r[metric + "_mean"])
            for tier in ("best", "runner_up"):
                if not remaining:
                    break
                anchor = Decimal(str(remaining[0][metric + "_mean"]))
                tolerance = max(Decimal("1e-12"), abs(anchor) * Decimal("1e-8"))
                tied = [r for r in remaining if Decimal(str(r[metric + "_mean"])) - anchor <= tolerance]
                for row in tied:
                    awards.append(dict(dataset=d, horizon=h, metric=metric, tier=tier, system=row["system"],
                                       tied=len(tied) > 1, mean=row[metric + "_mean"]))
                remaining = remaining[len(tied):]
    counts = []
    for system in dict.fromkeys(r["system"] for r in table):
        best = [r for r in awards if r["system"] == system and r["tier"] == "best"]
        counts.append(dict(system=system, MSE_best=sum(r["metric"] == "mse" for r in best),
                           MAE_best=sum(r["metric"] == "mae" for r in best), combined_best=len(best)))
    return awards, counts


def write_markdown(path, table, awards):
    systems = list(dict.fromkeys(row["system"] for row in table))
    units = list(dict.fromkeys((row["dataset"], row["horizon"]) for row in table))
    values = {(row["dataset"], row["horizon"], row["system"]): row for row in table}
    best = {(r["dataset"], r["horizon"], r["metric"], r["system"]) for r in awards if r["tier"] == "best"}
    body = ["# DEV MSE and MAE", "", "Five-seed means. Sample SD and full precision are retained in the CSV.", "",
            "| Dataset | Horizon | Metric | " + " | ".join(systems) + " |",
            "|---|---:|---|" + "---:|" * len(systems)]
    for dataset, horizon in units:
        for metric in ("mse", "mae"):
            cells = []
            for system in systems:
                row = values.get((dataset, horizon, system))
                if row is None:
                    raise ValueError("Cannot render an incomplete system/unit table")
                text = format(row[metric + "_mean"], ".6f")
                if (dataset, horizon, metric, system) in best:
                    text = "**" + text + "**"
                cells.append(text)
            body.append("| %s | %s | %s | %s |" % (dataset, horizon, metric.upper(), " | ".join(cells)))
    Path(path).write_text("\n".join(body) + "\n", encoding="utf8")


def pair(rows, treatment="HOVER", control="SIMPLE"):
    index = {(r["dataset"], r["horizon"], r["system"], r["seed"]): r for r in rows}
    if len(index) != len(rows):
        raise ValueError("Duplicate paired record")
    seed_rows, summary = [], []
    units = dict.fromkeys((r["dataset"], r["horizon"]) for r in rows if r["system"] == treatment)
    for d, h in units:
        paired = []
        for seed in range(951, 956):
            left, right = index[d, h, treatment, seed], index[d, h, control, seed]
            if (left["role"], right["role"]) != ("DEV", "DEV") or left["TEST_access"] or right["TEST_access"]:
                raise ValueError("Invalid paired roles")
            if not left.get("base_checkpoint_SHA256") or left["base_checkpoint_SHA256"] != right.get("base_checkpoint_SHA256"):
                raise ValueError("Same-seed frozen backbone identity mismatch")
            if left["source_version"] != right["source_version"]:
                raise ValueError("Paired configuration versions differ")
            if left["metrics"]["valid_count"] != right["metrics"]["valid_count"]:
                raise ValueError("Paired target count mismatch")
            row = dict(dataset=d, horizon=h, seed=seed, treatment=treatment, control=control, role="DEV")
            for metric in ("mse", "mae"):
                row[metric + "_difference"] = left["metrics"][metric] - right["metrics"][metric]
            paired.append((row, left, right))
            seed_rows.append(row)
        row = dict(dataset=d, horizon=h, treatment=treatment, control=control, role="DEV")
        for metric in ("mse", "mae"):
            diffs = [r[0][metric + "_difference"] for r in paired]
            baseline = statistics.mean(r[2]["metrics"][metric] for r in paired)
            row[metric + "_difference_mean"] = statistics.mean(diffs)
            row[metric + "_difference_SD"] = statistics.stdev(diffs)
            row[metric + "_gain_percent"] = -100 * statistics.mean(diffs) / baseline
            row[metric + "_winning_seeds"] = sum(v < 0 for v in diffs)
        summary.append(row)
    return seed_rows, summary


def plot_table(csv_path, output, dataset, metric="mse"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = list(csv.DictReader(Path(csv_path).open(encoding="utf8", newline="")))
    rows = [r for r in rows if r["dataset"] == dataset]
    if not rows or len({r["horizon"] for r in rows}) != 1:
        raise ValueError("Select one dataset and one horizon for this plot")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar([r["system"] for r in rows], [float(r[metric + "_mean"]) for r in rows],
           yerr=[float(r[metric + "_SD"]) for r in rows], capsize=3)
    ax.set_ylabel("DEV " + metric.upper())
    ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
