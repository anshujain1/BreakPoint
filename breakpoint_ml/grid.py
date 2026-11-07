"""Multi-seed benchmark grid: models x datasets x seeds, one JSON file per cell.

A *cell* is (dataset, model, seed). The seed controls the train/val/test split, the model's
random init, and the search RNG. For the synthetic datasets the data itself is fixed.

Cells are independent, so the grid runs in parallel, is resumable (finished cells are skipped),
and records failures instead of hiding them. Cells are ordered seed-major so an interrupted run
still has complete seeds.
"""
from __future__ import annotations

import json
import os
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from typing import Dict, List, Optional, Sequence

import numpy as np

SEARCHERS = ("random", "bandit", "surrogate")


def cell_path(out_dir: str, dataset: str, model: str, seed: int) -> str:
    return os.path.join(out_dir, "cells", f"{dataset}__{model}__{seed}.json")


def _cap_rows(X, y, cap: int, seed: int):
    if len(y) <= cap:
        return X, y
    from sklearn.model_selection import train_test_split
    _, Xs, _, ys = train_test_split(X, y, test_size=cap, stratify=y, random_state=seed)
    return Xs, ys


def run_cell(dataset: str, model: str, seed: int, budget: int = 100, tau: float = 0.10,
             ref_evals: tuple = (600, 150), rounds: int = 1, val_cap: int = 800, test_cap: int = 1000,
             fix_kwargs: Optional[dict] = None) -> dict:
    """Run one cell: strategy comparison on the baseline model, then the verified fix loop."""
    from .benchmark import reference_search, run_benchmark
    from .builtin import load_splits, make_model
    from .evaluator import Evaluator
    from .fix import find_and_fix
    from .space import SearchSpace
    from .strategies import make_strategy

    warnings.filterwarnings("ignore")
    t0 = time.time()
    space = SearchSpace()
    splits = load_splits(dataset, seed)
    # speed: search/gate and test measurements use at most val_cap / test_cap rows (stratified);
    # training always uses the full train split.
    Xv, yv = _cap_rows(splits.X_val, splits.y_val, val_cap, seed)
    Xt, yt = _cap_rows(splits.X_te, splits.y_te, test_cap, seed)
    splits = replace(splits, X_val=Xv, y_val=yv, X_te=Xt, y_te=yt)
    fitted = make_model(model, seed).fit(splits.X_tr, splits.y_tr)
    ev = Evaluator(fitted, splits.X_val, splits.y_val, tau=tau, seed=seed)

    ref = reference_search(ev, space, n_random=ref_evals[0], n_refine=ref_evals[1], seed=2024 + seed)
    bench = run_benchmark(ev, space, SEARCHERS, budget, [seed], ref)
    search = {
        n: dict(final_cost=float(o.final_costs[0]), n_failures=int(o.n_failures[0]),
                first_failure=int(o.first_failure[0]), evals_to_near_ref=int(o.evals_to_near_ref[0]),
                seconds=float(o.seconds[0]), curve=[float(v) for v in o.curves[0]])
        for n, o in bench.items()}

    kw = dict(budget=budget, rounds=rounds, tau=tau, seed=seed, battery_size=240, gate_size=200,
              members=5, reference_evals=None)
    kw.update(fix_kwargs or {})
    fix = find_and_fix(lambda: make_model(model, seed), splits, make_strategy("surrogate"), space, **kw)

    return dict(
        ok=True, dataset=dataset, model=model, seed=seed, budget=budget, tau=tau,
        n_train=int(len(splits.y_tr)), n_val=int(len(splits.y_val)), n_test=int(len(splits.y_te)), n_features=int(splits.X_tr.shape[1]),
        n_classes=int(len(np.unique(splits.y_tr))), clean_val=float(ev.clean_score),
        ref=dict(min_cost=float(ref.min_cost), fail_fraction=float(ref.fail_fraction), n_evals=int(ref.n_evals)),
        search=search, fix=dict(rounds=[asdict(r) for r in fix.rounds]),
        seconds=time.time() - t0)


def _worker(args) -> dict:
    out_dir, dataset, model, seed, kw = args
    path = cell_path(out_dir, dataset, model, seed)
    try:
        res = run_cell(dataset, model, seed, **kw)
    except Exception as e:  # recorded, never silent
        res = dict(ok=False, dataset=dataset, model=model, seed=seed, error=f"{type(e).__name__}: {e}",
                   trace=traceback.format_exc(limit=6))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f)
    os.replace(tmp, path)
    return res


def run_grid(models: Sequence[str], datasets: Sequence[str], seeds: Sequence[int], out_dir: str,
             workers: int = 2, log=print, **cell_kwargs) -> List[dict]:
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    todo, done = [], 0
    for sd in seeds:                       # seed-major
        for ds in datasets:
            for m in models:
                if os.path.exists(cell_path(out_dir, ds, m, sd)):
                    done += 1
                else:
                    todo.append((out_dir, ds, m, sd, cell_kwargs))
    total = len(todo) + done
    log(f"grid: {total} cells, {done} already done, {len(todo)} to run on {workers} workers")
    results, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_worker, a): a for a in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            tag = f"{r['dataset']}/{r['model']}/seed{r['seed']}"
            status = f"{r['seconds']:.0f}s" if r.get("ok") else f"FAILED {r['error']}"
            log(f"[{done + i}/{total}] {tag} {status} (elapsed {(time.time() - t0) / 60:.1f} min)")
    return results


def load_cells(out_dir: str) -> List[dict]:
    d = os.path.join(out_dir, "cells")
    if not os.path.isdir(d):
        return []
    cells = []
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".json"):
            with open(os.path.join(d, fn)) as f:
                cells.append(json.load(f))
    return cells
