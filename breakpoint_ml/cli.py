"""Command line: `python -m breakpoint_ml demo` or `... run --dataset wine --model svm`."""
from __future__ import annotations

import argparse
import os
import time
import warnings

from .benchmark import reference_search, run_benchmark
from .builtin import MODELS, dataset_names, load_csv_splits, load_splits, make_model
from .evaluator import Evaluator
from .fix import find_and_fix
from .report import write_report
from .space import SearchSpace
from .strategies import STRATEGIES, make_strategy

DEMO_COMBOS = [("breast_cancer", "rf"), ("wine", "svm")]
GRID_MODELS = ["logreg", "rf", "xgboost", "lightgbm", "catboost", "mlp"]
GRID_DATASETS = ["breast_cancer", "wine", "digits", "synthetic_large", "nonlinear", "multiclass"]


def run_combo(dataset: str, model: str, budget: int, seeds: int, rounds: int, tau: float,
              strategy: str, seed: int, log=print, splits=None) -> dict:
    log(f"== {dataset} / {model}")
    splits = splits if splits is not None else load_splits(dataset, seed)
    space = SearchSpace()
    fitted = make_model(model, seed).fit(splits.X_tr, splits.y_tr)
    ev = Evaluator(fitted, splits.X_val, splits.y_val, tau=tau, seed=seed)
    t = time.time()
    ref = reference_search(ev, space)
    log(f"reference search: cheapest failure {ref.min_cost:.3f} ({ref.n_evals} evals, {time.time() - t:.0f}s)")
    bench = run_benchmark(ev, space, list(STRATEGIES), budget, range(seeds), ref)
    for n, o in bench.items():
        log(f"  {n:10s} cheapest failure {o.final_costs.mean():.3f} +- {o.final_costs.std():.3f}")
    fix = find_and_fix(lambda: make_model(model, seed), splits, make_strategy(strategy), space,
                       budget=budget, rounds=rounds, tau=tau, seed=seed, log=lambda m: log("  " + m))
    return dict(dataset=dataset, model=model, budget=budget, ref=ref, bench=bench, fix=fix)


def main(argv=None):
    warnings.filterwarnings("ignore")
    p = argparse.ArgumentParser(prog="breakpoint_ml", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--budget", type=int, default=100, help="model evaluations per search")
    common.add_argument("--seeds", type=int, default=5, help="seeds for the strategy comparison")
    common.add_argument("--rounds", type=int, default=2, help="find-and-fix rounds")
    common.add_argument("--tau", type=float, default=0.10, help="accuracy drop that counts as failure")
    common.add_argument("--strategy", default="surrogate", choices=sorted(STRATEGIES))
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--out", default="runs/latest")
    d = sub.add_parser("demo", parents=[common], help="run the built-in demo combos")
    r = sub.add_parser("run", parents=[common], help="stress-test one dataset/model")
    r.add_argument("--dataset", choices=dataset_names(), help="built-in dataset")
    r.add_argument("--csv", help="path to your own CSV (use with --target)")
    r.add_argument("--target", help="target column name in the CSV")
    r.add_argument("--model", required=True, choices=sorted(MODELS))
    g = sub.add_parser("grid", help="multi-seed grid over models x datasets x seeds, then aggregate")
    g.add_argument("--models", nargs="+", default=GRID_MODELS, choices=sorted(MODELS))
    g.add_argument("--datasets", nargs="+", default=GRID_DATASETS, choices=dataset_names())
    g.add_argument("--seeds", type=int, default=5, help="number of seeds (0..n-1)")
    g.add_argument("--budget", type=int, default=100)
    g.add_argument("--tau", type=float, default=0.10)
    g.add_argument("--workers", type=int, default=2)
    g.add_argument("--out", default="runs/grid")
    ag = sub.add_parser("aggregate", help="(re)build grid_report.md from finished cells")
    ag.add_argument("--out", default="runs/grid")
    a = p.parse_args(argv)

    if a.cmd == "grid":
        from .aggregate import write_summary
        from .grid import run_grid
        run_grid(a.models, a.datasets, list(range(a.seeds)), a.out, workers=a.workers,
                 budget=a.budget, tau=a.tau)
        print("summary written to", write_summary(a.out))
        return
    if a.cmd == "aggregate":
        from .aggregate import write_summary
        print("summary written to", write_summary(a.out))
        return

    if a.cmd == "demo":
        combos = [(ds, m, None) for ds, m in DEMO_COMBOS]
    elif a.csv:
        if not a.target:
            p.error("--csv needs --target")
        combos = [(os.path.splitext(os.path.basename(a.csv))[0], a.model,
                   load_csv_splits(a.csv, a.target, a.seed))]
    elif a.dataset:
        combos = [(a.dataset, a.model, None)]
    else:
        p.error("run needs --dataset or --csv")
    sections = [run_combo(ds, m, a.budget, a.seeds, a.rounds, a.tau, a.strategy, a.seed, splits=sp)
                for ds, m, sp in combos]
    path = write_report(sections, a.out, dict(tau=a.tau, budget=a.budget, seeds=a.seeds))
    print(f"report written to {path}")


if __name__ == "__main__":
    main()
