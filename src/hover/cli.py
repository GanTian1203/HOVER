"""Command-line entry points for data, experiments, and results."""
import argparse
import json
import os
from pathlib import Path
import sys
import subprocess
import time

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(variable, "1")


def demo(output):
    import torch
    from torch import nn
    from .io import save
    from .method import HOVER, Simple, adapter_loss

    torch.set_num_threads(1)
    class Reference(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(336, 96)
        def forward(self, x, xmark=None, future_mark=None):
            return self.layer(x.transpose(1, 2)).transpose(1, 2)
    torch.manual_seed(7)
    base = Reference().eval()
    x, y = torch.randn(2, 336, 3), torch.randn(2, 96, 3)
    rows = []
    for name, cls in (("HOVER", HOVER), ("SIMPLE", Simple)):
        model = cls(base, 731)
        p, b, _ = model.components(x)
        assert torch.equal(p, b)
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.001)
        loss, primary, retention = adapter_loss(p, y, b, 1.)
        optimizer.zero_grad()
        loss.backward()
        assert all(p.grad is None for p in model.base.parameters())
        optimizer.step()
        rows.append(dict(system=name, parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                         input=list(x.shape), output=list(p.shape), initial_equals_base=True, frozen_base=True,
                         CPU_Adam_updates=1, synthetic_primary=float(primary), synthetic_retention=float(retention)))
    result = dict(status="PASS", kind="SYNTHETIC_CPU_EXAMPLE", rows=rows, real_dataset_fits=0, TEST_access=False)
    if output:
        save(output, result)
    return result


def parser():
    p = argparse.ArgumentParser(description="HOVER original-formal DEV reproduction")
    commands = p.add_subparsers(dest="command", required=True)
    sample = commands.add_parser("demo", help="Two synthetic CPU forward/backward examples")
    sample.add_argument("--output")
    technical = commands.add_parser("smoke", help="Full-batch real-window technical step and recovery, not a formal FIT")
    technical.add_argument("--dataset",required=True)
    technical.add_argument("--data-root",required=True)
    technical.add_argument("--output",required=True)
    technical.add_argument("--backbone",choices=("NLINEAR_PAIRED","OLINEAR"),default="NLINEAR_PAIRED")
    technical.add_argument("--horizon",type=int,default=96)
    technical.add_argument("--device",default="cpu")
    technical.add_argument("--external-root",default="external_models")
    data = commands.add_parser("prepare", help="Prepare only the fixed TRAIN/DEV numeric prefix")
    data.add_argument("--dataset", required=True)
    data.add_argument("--source", required=True)
    data.add_argument("--output", required=True)
    download = commands.add_parser("download", help="Download only a pre-audited TRAIN/DEV byte range; reject whole-file fallback")
    download.add_argument("--dataset", required=True)
    download.add_argument("--output", required=True)
    sources = commands.add_parser("sources", help="Acquire exact official comparator source files")
    sources.add_argument("--model", required=True)
    sources.add_argument("--output", default="external_models")
    sources.add_argument("--local-path", help="Register a separately obtained read-only official source tree")
    sources.add_argument("--commit", help="Explicit immutable commit for local-source registration")
    sources.add_argument("--permission-basis", help="Actual documented use basis;required for OLinear")
    sources.add_argument("--verify", action="store_true", help="Verify already acquired/registered source without network access")
    run = commands.add_parser("run", help="Plan or execute A/B/C/Table2; no fit starts without --execute")
    run.add_argument("--stage", choices=("A", "B", "C", "Table2"), required=True)
    run.add_argument("--dataset")
    run.add_argument("--system")
    run.add_argument("--seed", type=int)
    run.add_argument("--horizon", type=int)
    run.add_argument("--data-root", default="data/prepared")
    run.add_argument("--output", default="outputs/reproduction")
    run.add_argument("--external-root", default="external_models")
    run.add_argument("--device", default="cpu")
    run.add_argument("--threads", type=int, default=1)
    run.add_argument("--execute", action="store_true")
    run.add_argument("--worker-one", action="store_true", help=argparse.SUPPRESS)
    run.add_argument("--result-json", help=argparse.SUPPRESS)
    summary = commands.add_parser("summarize", help="Aggregate standardized DONE DEV records")
    summary.add_argument("--records")
    summary.add_argument("--stage")
    summary.add_argument("--reference-csv")
    summary.add_argument("--reference-json")
    summary.add_argument("--kind", choices=("main", "ablation"), default="main")
    summary.add_argument("--output", required=True)
    summary.add_argument("--treatment")
    summary.add_argument("--control")
    summary.add_argument("--dataset")
    summary.add_argument("--horizon", type=int)
    plot = commands.add_parser("plot", help="Plot numeric summary CSV using mean and sample SD")
    plot.add_argument("--table", required=True)
    plot.add_argument("--dataset", required=True)
    plot.add_argument("--metric", choices=("mse", "mae"), default="mse")
    plot.add_argument("--output", required=True)
    audit = commands.add_parser("audit", help="Independently recheck checkpoint and numeric aggregates")
    audit.add_argument("--fit-dir", required=True)
    audit.add_argument("--data-root")
    audit.add_argument("--base-fit-dir")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    from .io import Blocked, save
    try:
        if args.command == "demo":
            result = demo(args.output)
        elif args.command == "prepare":
            from .prepare import prepare
            result = prepare(args.dataset, args.source, args.output)
        elif args.command == "download":
            from .prepare import download
            result = download(args.dataset, args.output)
        elif args.command == "sources":
            from .sources import acquire, register, verify
            if args.local_path:
                result = register(args.model,args.local_path,args.output,args.commit,args.permission_basis)
            elif args.verify:
                result = verify(args.model,args.output)
            else:
                result = acquire(args.model, args.output)
        elif args.command == "run":
            from .experiments import Experiment, task_matrix
            tasks = task_matrix(args.stage, args.dataset, args.system, args.seed, args.horizon)
            if not args.execute:
                result = dict(status="PLAN_ONLY", formal_records=len(tasks), tasks=tasks,
                              dependencies="native, role-specific Q if needed, inner731/732, selection, formal checkpoint", TEST_access=False)
            else:
                completed, blocked = [], []
                if len(tasks) > 1:
                    if args.worker_one:
                        raise ValueError("A task worker must resolve to one formal record")
                    # Separate processes isolate conflicting upstream module names.
                    from .io import load
                    for task in tasks:
                        log = Path(args.output) / "matrix_logs" / args.stage / task["dataset"] / str(task["horizon"]) / task["system"]
                        log.mkdir(parents=True, exist_ok=True)
                        log = log / str(task["seed"]) / ("attempt_" + str(time.time_ns()))
                        log.mkdir(parents=True, exist_ok=False)
                        stem = log / "worker"
                        result_path = stem.with_suffix(".json")
                        command = [sys.executable, "-B", "-m", "hover", "run", "--stage", args.stage,
                            "--dataset", task["dataset"], "--horizon", str(task["horizon"]), "--system", task["system"],
                            "--seed", str(task["seed"]), "--data-root", args.data_root, "--output", args.output,
                            "--external-root", args.external_root, "--device", args.device, "--threads", str(args.threads),
                            "--execute", "--worker-one", "--result-json", str(result_path)]
                        with stem.with_suffix(".stdout.log").open("w", encoding="utf8") as stdout, stem.with_suffix(".stderr.log").open("w", encoding="utf8") as stderr:
                            process = subprocess.run(command, stdout=stdout, stderr=stderr)
                        child = load(result_path) if result_path.exists() else dict(status="BLOCKED", error="Worker exited before a result receipt", exit_code=process.returncode)
                        if process.returncode == 0 and child["status"] == "DONE":
                            completed.append(task)
                        else:
                            blocked.append(dict(task=task, worker_result=child, exit_code=process.returncode))
                else:
                    import torch
                    torch.set_num_threads(args.threads)
                    engine = Experiment(args.data_root, args.output, args.device, args.external_root)
                    for task in tasks:
                        try:
                            completed.append(engine.run(task))
                        except (Blocked, ValueError, OSError, RuntimeError) as error:
                            failure = dict(task=task, status="BLOCKED", error=str(error), TEST_access=False)
                            blocked.append(failure)
                            save(Path(args.output) / "blocked" / args.stage / task["dataset"] / str(task["horizon"]) /
                                 task["system"] / (str(task["seed"]) + ".json"), failure)
                result = dict(status="PARTIAL" if blocked else "DONE", completed_records=len(completed), blocked=blocked, TEST_access=False)
        elif args.command == "smoke":
            import torch
            torch.set_num_threads(1)
            from .smoke import smoke
            result=smoke(args.dataset,args.data_root,args.output,args.backbone,args.horizon,args.device,args.external_root)
        elif args.command == "summarize":
            from .analysis import pair, placements, read_records, records_from_csv, summarize, write_csv, write_markdown
            if sum(bool(value) for value in (args.records, args.reference_csv, args.reference_json)) != 1:
                raise ValueError("Choose records, reference-csv or reference-json")
            if args.records and args.stage not in ("A", "B", "C", "Table2"):
                raise ValueError("Select one frozen --stage when reading experiment records")
            if args.reference_csv:
                rows = records_from_csv(args.reference_csv, args.kind)
            elif args.reference_json:
                from .io import load
                rows = load(args.reference_json)
                if not isinstance(rows, list):
                    raise ValueError("Reference JSON must contain standardized record objects")
            else:
                rows = read_records(args.records, args.stage)
            if args.dataset:
                rows = [r for r in rows if r["dataset"] == args.dataset]
            if args.horizon is not None:
                rows = [r for r in rows if r["horizon"] == args.horizon]
            table = summarize(rows)
            awards, counts = placements(table)
            out = Path(args.output)
            write_csv(out / "DEV_summary.csv", table)
            write_csv(out / "DEV_placements.csv", awards)
            write_csv(out / "DEV_metric_best_count.csv", counts)
            write_markdown(out / "DEV_table.md", table, awards)
            if args.treatment or args.control:
                if not args.treatment or not args.control:
                    raise ValueError("Pair requires both treatment and control")
                seeds, groups = pair(rows, args.treatment, args.control)
                write_csv(out / "DEV_paired_seeds.csv", seeds)
                write_csv(out / "DEV_paired_summary.csv", groups)
            if args.kind == "ablation" or args.stage == "B":
                from .experiments import CONTROLS
                all_groups, wins = [], []
                for control in CONTROLS:
                    _, groups = pair(rows, "HOVER", control)
                    all_groups.extend(groups)
                    for family in ("all", "ETT", "PEMS", "other_LTF"):
                        selected = [r for r in groups if family == "all" or
                                    (family == "ETT" and r["dataset"].startswith("ETT")) or
                                    (family == "PEMS" and r["dataset"].startswith("PEMS")) or
                                    (family == "other_LTF" and not r["dataset"].startswith(("ETT", "PEMS")))]
                        wins.append(dict(control=control, family=family, tasks=len(selected),
                            MSE_winning_tasks=sum(r["mse_difference_mean"] < 0 for r in selected),
                            MAE_winning_tasks=sum(r["mae_difference_mean"] < 0 for r in selected)))
                write_csv(out / "DEV_all_control_pairs.csv", all_groups)
                write_csv(out / "DEV_control_task_wins.csv", wins)
            result = dict(status="DONE", groups=len(table), seed_records=len(rows), role="DEV", TEST_access=False)
        elif args.command == "audit":
            from .audit import audit_fit
            result = audit_fit(args.fit_dir, args.data_root, args.base_fit_dir)
        else:
            from .analysis import plot_table
            plot_table(args.table, args.output, args.dataset, args.metric)
            result = dict(status="DONE", figure=args.output, role="DEV")
        if getattr(args, "result_json", None):
            save(args.result_json, result)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 2 if result["status"] in ("PARTIAL", "BLOCKED") else 0
    except (Blocked, ValueError, OSError, KeyError) as error:
        print(json.dumps(dict(status="BLOCKED", error=str(error), TEST_access=False)), file=sys.stderr)
        raise SystemExit(2)
