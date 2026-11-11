"""Reconcile the manuscript cells and held-out aggregates with released CSVs."""
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent.parent
SUPP = ROOT / "supplementary"
TABLES = ROOT / "results/tables"
SEEDS = set(range(951, 956))


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def keyed(path, *fields):
    rows = read(path)
    result = {tuple(row[f] for f in fields): row for row in rows}
    assert len(result) == len(rows), path
    return result


def close(actual, expected):
    assert math.isclose(float(actual), expected, rel_tol=1e-10, abs_tol=1e-13), (actual, expected)


def table_rows(name):
    text = (SUPP / "paper-tables" / name).read_text(encoding="utf8")
    dataset = ""
    for line in text.splitlines():
        if "&" not in line or not re.search(r"\d\.\d{5}", line):
            continue
        fields = line.split("&")
        dataset = fields[0].strip() or dataset
        yield dataset, fields[1].strip(), re.findall(r"(?<![\d.])\d+\.\d{5}(?!\d)", line)


def values(row):
    return [float(row["mse_mean"]), float(row["mae_mean"])]


def check_display(label, cells, expected):
    assert cells == [f"{value:.5f}" for value in expected], (label, cells, expected)
    return len(cells)


def paper_cells():
    total = 0
    main = keyed(TABLES / "Table1_MSE_MAE_mean_SD.csv", "dataset", "system")
    systems = ("OLINEAR_HOVER", "NLINEAR_HOVER", "OLINEAR", "PATCHTST",
               "ITRANSFORMER", "SIGMA", "TIMEMIXER", "NLINEAR_STRONG", "DLINEAR", "FITS")
    for dataset, _, cells in table_rows("table1.tex"):
        total += check_display(dataset, cells, [v for arm in systems for v in values(main[dataset, arm])])
    cross = keyed(TABLES / "Table2_three_arm_MSE_MAE_mean_SD.csv", "dataset", "system")
    for name, arms in [("table2.tex", ("FROZEN", "HOVER")), ("table3.tex", ("FROZEN", "SIMPLE", "HOVER"))]:
        for dataset, backbone, cells in table_rows(name):
            model = {"NLinear": "NLINEAR_PAIRED", "iTrans.": "ITRANSFORMER"}.get(backbone, backbone.upper())
            total += check_display((name, dataset, model), cells, [v for arm in arms for v in values(cross[dataset, model+"+"+arm])])
    ablation = keyed(TABLES / "Table3_MSE_MAE_mean_SD.csv", "dataset", "system")
    controls = ("HOVER", "history_only", "no_history_direct", "concatenation", "history_condition_replacement")
    for dataset, _, cells in table_rows("table4.tex"):
        total += check_display(dataset, cells, [v for arm in controls for v in values(ablation[dataset, arm])])
    horizons = keyed(ROOT / "results/horizon_final/DEV_48_system_groups.csv", "dataset", "horizon", "system")
    for dataset, horizon, cells in table_rows("table5.tex"):
        total += check_display((dataset, horizon), cells, [v for arm in ("FROZEN", "HOVER") for v in values(horizons[dataset, horizon, arm])])
    assert total == 520, total
    return total


def holdout():
    data = read(SUPP / "test/TEST_system_metrics.csv")
    summary = keyed(SUPP / "test/TEST_five_seed_summary.csv", "dataset", "backbone")
    paired = keyed(SUPP / "test/TEST_paired_seeds.csv", "dataset", "backbone", "seed")
    groups = defaultdict(dict)
    assert len(data) == 80 and len(paired) == 40 and len(summary) == 8
    for row in data:
        assert row["role"] == "TEST"
        for metric, aggregate in [("MSE", "SSE"), ("MAE", "SAE")]:
            close(row[metric], float(row[aggregate]) / int(row["valid_count"]))
        key = row["dataset"], row["backbone"]
        pair = groups[key].setdefault(int(row["seed"]), {})
        assert row["system"] not in pair
        pair[row["system"]] = row
    for key, seeds in groups.items():
        assert set(seeds) == SEEDS
        for seed, rows in seeds.items():
            assert set(rows) == {"Frozen", "HOVER"}
            for field in ("base_checkpoint_SHA256", "source_SHA256", "train_scaler_SHA256", "valid_count", "first_origin", "last_origin"):
                assert rows["Frozen"][field] == rows["HOVER"][field], (key, seed, field)
            for metric in ("MSE", "MAE"):
                row = paired[key + (str(seed),)]
                for arm in rows:
                    close(row[f"{arm}_{metric}"], float(rows[arm][metric]))
                close(row[f"{metric}_delta"], float(rows["HOVER"][metric]) - float(rows["Frozen"][metric]))
        for metric in ("MSE", "MAE"):
            frozen, hover = ([float(rows[arm][metric]) for rows in seeds.values()] for arm in ("Frozen", "HOVER"))
            for arm, series in [("Frozen", frozen), ("HOVER", hover)]:
                close(summary[key][f"{arm}_{metric}_mean"], mean(series))
                close(summary[key][f"{arm}_{metric}_sample_SD"], stdev(series))
            close(summary[key][f"{metric}_relative_reduction_pct"], 100 * (1 - mean(hover) / mean(frozen)))
    total = 0
    for dataset, model, cells in table_rows("table_test.tex"):
        row = summary[dataset, "NLINEAR_PAIRED" if model == "NLinear" else model.upper()]
        total += check_display((dataset, model), cells, [float(row[f"{arm}_{m}_mean"]) for m in ("MSE", "MAE") for arm in ("Frozen", "HOVER")])
    assert total == 32
    return total


def posty():
    dev = read(SUPP / "posty/DEV_seed_results.csv")
    dev_summary = keyed(SUPP / "posty/DEV_five_seed_summary.csv", "dataset", "system", "metric")
    test = read(SUPP / "posty/FIG7_DLINEAR_TEST_RESULTS.csv")
    test_summary = keyed(SUPP / "posty/FIG7_DLINEAR_TEST_FIVE_SEED_SUMMARY.csv", "dataset", "system")
    assert len(dev) == 10 and len(test) == 40
    for dataset, expected in [("ECL", (4.63, 2.84)), ("PEMS03", (30.86, 13.42))]:
        rows = [r for r in dev if r["dataset"] == dataset]
        assert {int(r["seed"]) for r in rows} == SEEDS
        for metric, gain in zip(("MSE", "MAE"), expected):
            for arm in ("Frozen", "Delta", "HOVER"):
                vals = [float(r[f"{arm}_{metric}"]) for r in rows]
                close(dev_summary[dataset, arm, metric]["mean"], mean(vals))
                close(dev_summary[dataset, arm, metric]["sample_sd"], stdev(vals))
            delta, hover = ([float(r[f"{a}_{metric}"]) for r in rows] for a in ("Delta", "HOVER"))
            assert round(100 * (1 - mean(hover) / mean(delta)), 2) == gain
            if metric == "MSE":
                assert all(h < p for h, p in zip(hover, delta))
        for arm in ("Frozen", "Simple", "PostY", "HOVER"):
            rows = [r for r in test if r["dataset"] == dataset and r["system"] == arm]
            assert {int(r["seed"]) for r in rows} == SEEDS
            for metric, aggregate in [("MSE", "SSE"), ("MAE", "SAE")]:
                for row in rows:
                    assert row["role"] == "TEST"
                    close(row[metric], float(row[aggregate]) / int(row["valid_count"]))
                vals = [float(r[metric]) for r in rows]
                close(test_summary[dataset, arm][f"{metric}_mean"], mean(vals))
                close(test_summary[dataset, arm][f"{metric}_sample_SD"], stdev(vals))


if __name__ == "__main__":
    dev_cells, test_cells = paper_cells(), holdout()
    posty()
    print(f"Verified {dev_cells} DEV and {test_cells} TEST manuscript cells, 10 PostY DEV pairs, and 40 DLinear TEST system scores.")


