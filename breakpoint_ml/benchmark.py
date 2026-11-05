"""Compare search strategies against a high-budget reference search."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from .evaluator import Evaluator, Result
from .space import Config, SearchSpace
from .strategies import make_strategy

NO_FAILURE_COST = 1.0   # running-min cost reported while no failure has been found yet


def breakpoint_curve(history: Sequence[Result], n: int) -> np.ndarray:
    """Running minimum cost over failing configs after 1..n evaluations (1.0 = none yet)."""
    out = np.full(n, NO_FAILURE_COST)
    best = NO_FAILURE_COST
    for i, r in enumerate(history[:n]):
        if r.failed:
            best = min(best, r.config.cost)
        out[i] = best
    if len(history) < n:
        out[len(history):] = best
    return out


@dataclass
class Reference:
    """Best-effort ground truth: a big random sample plus local refinement of its best failures."""
    min_cost: float
    best_config: Config | None
    fail_fraction: float
    n_evals: int


def reference_search(ev: Evaluator, space: SearchSpace, n_random: int = 2500, n_refine: int = 500,
                     seed: int = 12345) -> Reference:
    ev = ev.fork()
    rng = np.random.default_rng(seed)
    for _ in range(n_random):
        ev.evaluate(space.sample(rng))
    fail_fraction = float(np.mean([r.failed for r in ev.history]))
    for _ in range(n_refine):
        fails = ev.failures()
        if not fails:
            break
        seed_cfg = fails[int(rng.integers(min(5, len(fails))))].config
        ev.evaluate(space.mutate(seed_cfg, rng, scale=0.05))
    fails = ev.failures()
    return Reference(fails[0].config.cost if fails else NO_FAILURE_COST,
                     fails[0].config if fails else None, fail_fraction, ev.n_evals)


@dataclass
class StrategyOutcome:
    name: str
    curves: np.ndarray                      # (seeds, budget) running-min failure cost
    seconds: List[float] = field(default_factory=list)
    n_failures: List[int] = field(default_factory=list)
    first_failure: List[int] = field(default_factory=list)
    evals_to_near_ref: List[int] = field(default_factory=list)

    @property
    def final_costs(self) -> np.ndarray:
        return self.curves[:, -1]


def run_benchmark(ev: Evaluator, space: SearchSpace, strategies: Sequence[str], budget: int,
                  seeds: Sequence[int], ref: Reference, near: float = 0.05) -> Dict[str, StrategyOutcome]:
    out: Dict[str, StrategyOutcome] = {}
    for name in strategies:
        curves, o = [], StrategyOutcome(name, np.empty((0, budget)))
        for sd in seeds:
            fork = ev.fork()
            t0 = time.time()
            hist = make_strategy(name).run(fork, space, budget, np.random.default_rng(1000 + sd))
            o.seconds.append(time.time() - t0)
            curve = breakpoint_curve(hist, budget)
            curves.append(curve)
            o.n_failures.append(sum(r.failed for r in hist[:budget]))
            hit = np.flatnonzero(curve < NO_FAILURE_COST)
            o.first_failure.append(int(hit[0]) + 1 if hit.size else budget + 1)
            near_hit = np.flatnonzero(curve <= ref.min_cost + near)
            o.evals_to_near_ref.append(int(near_hit[0]) + 1 if near_hit.size else budget + 1)
        o.curves = np.vstack(curves)
        out[name] = o
    return out
