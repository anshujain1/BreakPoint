"""Find-and-fix: search for failures, propose fixes, *verify* them, keep only what works.

A fix is never assumed to work. Each round:

  1. SEARCH     the current model is stress-tested on `val`; the cheapest failures are collected.
  2. DIAGNOSE   failures are summarized (which feature targets / perturbation families dominate).
  3. PROPOSE    several candidate fixes are trained:
                  targeted_augmentation  retrain on perturbed copies of the failures just found
                  random_augmentation    same data volume from blind random configs (the control)
                  feature_subspace       random-subspace ensemble: no member sees every feature
  4. VERIFY     each candidate is scored on a *validation* gate battery of gentle configs
                (failure rate) plus clean accuracy. A candidate is accepted only if it lowers the
                failure rate by `margin` without costing more than `max_clean_drop` clean accuracy.
  5. REPORT     the accepted model is measured on `test`, which nothing above ever touched.

Data discipline: train fits models, val drives search and the accept/reject gate, test is only
used for the reported before/after numbers. If no candidate passes the gate, the loop says so and
keeps the current model; that is a valid and honest outcome.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from sklearn.ensemble import BaggingClassifier

from .benchmark import Reference, reference_search
from .builtin import Splits
from .evaluator import Evaluator, Result
from .space import Config, SearchSpace
from .strategies import Strategy

LOW_COST = 0.25   # "gentle" perturbation: cost = severity x coverage <= this
CANDIDATES = ("targeted_augmentation", "random_augmentation", "feature_subspace")


# ---------------------------------------------------------------------------
# augmentation
# ---------------------------------------------------------------------------
def select_for_augmentation(failures: Sequence[Result], k: int) -> List[Config]:
    """Pick up to k failing configs, cheapest first, round-robin across families for diversity."""
    groups: Dict[str, List[Result]] = {}
    for r in sorted(failures, key=lambda r: r.config.cost):
        groups.setdefault(r.config.family, []).append(r)
    picked: List[Config] = []
    while len(picked) < k and any(groups.values()):
        for fam in sorted(groups, key=lambda f: groups[f][0].config.cost if groups[f] else 9):
            if groups[fam] and len(picked) < k:
                picked.append(groups[fam].pop(0).config)
    return picked


def make_augmentation(ev: Evaluator, X: np.ndarray, y: np.ndarray, configs: Sequence[Config],
                      space: SearchSpace, ratio: float, rng: np.random.Generator,
                      chunks: int = 8, sev_jitter=(0.5, 1.0)):
    """Perturbed copies of training rows, labels unchanged.

    Total new rows = ratio * len(X), split evenly over `configs`. Each config is applied in
    `chunks` separately-seeded pieces (so random column sets and shift directions vary) with
    severity x U(sev_jitter). The default jitter stays at or *below* the failing severity:
    training past the failure point teaches the model to ignore informative features.
    Column choices ('important'/'unimportant') are resolved with the model that exposed them.
    """
    if not configs:
        return np.empty((0, X.shape[1])), np.empty((0,), dtype=y.dtype)
    per_cfg = max(1, int(round(ratio * len(X) / len(configs))))
    per_chunk = max(1, per_cfg // chunks)
    xs, ys = [], []
    for ci, cfg in enumerate(configs):
        for ch in range(chunks):
            sev = float(np.clip(cfg.severity * rng.uniform(*sev_jitter), *space.sev_range))
            jit = Config.make(cfg.family, sev, cfg.coverage, cfg.target, salt=1 + ci * 97 + ch)
            rows = rng.integers(0, len(X), size=per_chunk)
            xs.append(ev.perturb(X[rows], jit))
            ys.append(y[rows])
    return np.vstack(xs), np.concatenate(ys)


def make_battery(space: SearchSpace, n: int, seed: int) -> List[Config]:
    rng = np.random.default_rng(seed)
    return [space.sample(rng) for _ in range(n)]


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------
@dataclass
class CandidateResult:
    name: str
    gate_failure_rate: float      # on val gate battery (lower is better)
    clean_val: float
    eligible: bool                # passed the accept gate
    accepted: bool = False


@dataclass
class RoundStats:
    round: int
    recipe: str                          # fixes accepted so far
    n_train: int
    clean_val: float
    clean_test: float
    breakpoint_cost: Optional[float]     # cheapest failure found by the search on val
    n_failures: int                      # failing configs among the search budget
    battery_mean_score: float            # mean test score over the fixed battery
    battery_failure_rate: float          # fraction of battery configs failing (test)
    battery_low_cost_failure_rate: float  # same, gentle configs only
    repair_rate: Optional[float] = None  # previous round's failures that no longer fail (test)
    diagnosis: Dict[str, float] = field(default_factory=dict)
    candidates: List[CandidateResult] = field(default_factory=list)
    decision: str = ""
    top_failures: List[dict] = field(default_factory=list)


@dataclass
class FixRun:
    rounds: List[RoundStats]
    tau: float
    budget: int
    strategy: str
    reference_before: Optional[Reference] = None   # independent big search on test, first model
    reference_after: Optional[Reference] = None    # ... and last model
    models: list = field(default_factory=list, repr=False)
    history_per_round: List[List[Result]] = field(default_factory=list, repr=False)


@dataclass
class _Recipe:
    aug_X: np.ndarray
    aug_y: np.ndarray
    subspace: bool = False
    steps: tuple = ()

    def describe(self) -> str:
        return " + ".join(self.steps) if self.steps else "baseline"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _fit(model_factory, splits: Splits, recipe: _Recipe, seed: int, members: int):
    X = np.vstack([splits.X_tr, recipe.aug_X])
    y = np.concatenate([splits.y_tr, recipe.aug_y])
    est = model_factory()
    if recipe.subspace:
        est = BaggingClassifier(estimator=est, n_estimators=members, max_features=0.5,
                                bootstrap=False, random_state=seed)
    return est.fit(X, y), len(X)


def diagnose(failures: Sequence[Result], k: int = 20) -> Dict[str, float]:
    """Share of the k cheapest failures per target and per family (keys 'target:x', 'family:y')."""
    top = list(failures)[:k]
    out: Dict[str, float] = {}
    for r in top:
        for key in (f"target:{r.config.target}", f"family:{r.config.family}"):
            out[key] = out.get(key, 0.0) + 1.0 / len(top)
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _top(failures: Sequence[Result], n: int = 8) -> List[dict]:
    return [dict(family=r.config.family, target=r.config.target, severity=r.config.severity,
                 coverage=r.config.coverage, cost=r.config.cost, drop=r.drop)
            for r in list(failures)[:n]]


def _gate(model, splits: Splits, gate_cfgs, tau, metric, seed):
    ev = Evaluator(model, splits.X_val, splits.y_val, tau=tau, metric=metric, seed=seed)
    return float(np.mean([ev.evaluate(c).failed for c in gate_cfgs])), ev.clean_score


def _measure(rnd, recipe, model, n_train, ev_val, splits, battery, tau, metric, seed,
             prev_failures=()) -> RoundStats:
    fails = ev_val.failures()
    ev_te = Evaluator(model, splits.X_te, splits.y_te, tau=tau, metric=metric, seed=seed)
    res = [ev_te.evaluate(c) for c in battery]
    low = [r.failed for r in res if r.config.cost <= LOW_COST]
    repair = None
    if len(prev_failures):
        repair = 1.0 - float(np.mean([ev_te.evaluate(c).failed for c in prev_failures]))
    return RoundStats(
        round=rnd, recipe=recipe.describe(), n_train=n_train, clean_val=ev_val.clean_score,
        clean_test=ev_te.clean_score,
        breakpoint_cost=fails[0].config.cost if fails else None, n_failures=len(fails),
        battery_mean_score=float(np.mean([r.score for r in res])),
        battery_failure_rate=float(np.mean([r.failed for r in res])),
        battery_low_cost_failure_rate=float(np.mean(low)) if low else 0.0,
        repair_rate=repair, diagnosis=diagnose(fails), top_failures=_top(fails))


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------
def find_and_fix(model_factory: Callable, splits: Splits, strategy: Strategy, space: SearchSpace,
                 budget: int = 150, rounds: int = 2, tau: float = 0.10, metric: str = "accuracy",
                 seed: int = 0, n_aug_configs: int = 20, aug_ratio: float = 0.5,
                 chunks: int = 8, sev_jitter=(0.5, 1.0), members: int = 10,
                 margin: float = 0.02, max_clean_drop: float = 0.015,
                 battery_size: int = 240, gate_size: int = 300,
                 reference_evals: Optional[tuple] = (1500, 300),
                 log: Callable[[str], None] = lambda s: None) -> FixRun:
    rng = np.random.default_rng(seed)
    battery = make_battery(space, battery_size, seed=777)            # test-side, reporting only
    gate_cfgs = [c for c in make_battery(space, 4 * gate_size, seed=555)   # val-side, accept/reject
                 if c.cost <= LOW_COST][:gate_size]

    recipe = _Recipe(np.empty((0, splits.X_tr.shape[1])), np.empty((0,), dtype=splits.y_tr.dtype))
    model, n_train = _fit(model_factory, splits, recipe, seed, members)
    stats: List[RoundStats] = []
    histories: List[List[Result]] = []
    models = [model]
    prev_failures: List[Config] = []

    for r in range(rounds + 1):
        ev_val = Evaluator(model, splits.X_val, splits.y_val, tau=tau, metric=metric, seed=seed)
        histories.append(strategy.run(ev_val, space, budget, np.random.default_rng(seed * 100 + r)))
        st = _measure(r, recipe, model, n_train, ev_val, splits, battery, tau, metric, seed, prev_failures)
        stats.append(st)
        log(f"round {r} [{st.recipe}]: clean_test={st.clean_test:.3f} breakpoint={st.breakpoint_cost} "
            f"low-cost battery failures={st.battery_low_cost_failure_rate:.3f}")
        if r == rounds:
            break

        fails = ev_val.failures()
        prev_failures = [res.config for res in fails[:50]]
        if not fails:
            st.decision = "No failures found within the budget; nothing to fix."
            log("  " + st.decision)
            break

        # --- propose -------------------------------------------------------
        proposals: Dict[str, tuple] = {}
        chosen = select_for_augmentation(fails, n_aug_configs)
        Xt, yt = make_augmentation(ev_val, splits.X_tr, splits.y_tr, chosen, space, aug_ratio, rng,
                                   chunks, sev_jitter)
        proposals["targeted_augmentation"] = _Recipe(
            np.vstack([recipe.aug_X, Xt]), np.concatenate([recipe.aug_y, yt]), recipe.subspace,
            recipe.steps + (f"targeted_augmentation(r{r + 1})",))
        rand_cfgs = [space.sample(rng) for _ in range(len(chosen))]
        Xr, yr = make_augmentation(ev_val, splits.X_tr, splits.y_tr, rand_cfgs, space, aug_ratio,
                                   rng, chunks, sev_jitter)
        proposals["random_augmentation"] = _Recipe(
            np.vstack([recipe.aug_X, Xr]), np.concatenate([recipe.aug_y, yr]), recipe.subspace,
            recipe.steps + (f"random_augmentation(r{r + 1})",))
        if not recipe.subspace:
            proposals["feature_subspace"] = _Recipe(recipe.aug_X, recipe.aug_y, True,
                                                    recipe.steps + ("feature_subspace",))

        # --- verify --------------------------------------------------------
        base_rate, base_clean = _gate(model, splits, gate_cfgs, tau, metric, seed)
        # A clean-accuracy tolerance below one validation sample is below the metric's resolution
        # (1 sample = 2.3 points on a 44-row validation set), so never ask for less than that.
        clean_tol = max(max_clean_drop, 1.0 / len(splits.y_val) + 1e-9)
        fitted, cands = {}, []
        for name in CANDIDATES:
            if name not in proposals:
                continue
            cand_model, cand_n = _fit(model_factory, splits, proposals[name], seed + r + 1, members)
            rate, clean = _gate(cand_model, splits, gate_cfgs, tau, metric, seed)
            ok = (rate <= base_rate - margin) and (clean >= base_clean - clean_tol)
            cands.append(CandidateResult(name, rate, clean, ok))
            fitted[name] = (cand_model, cand_n)
            log(f"  candidate {name}: gate failures {base_rate:.3f} -> {rate:.3f}, "
                f"clean_val {base_clean:.3f} -> {clean:.3f}, {'PASS' if ok else 'reject'}")
        eligible = [c for c in cands if c.eligible]
        st.candidates = cands
        if eligible:
            best = min(eligible, key=lambda c: (c.gate_failure_rate, CANDIDATES.index(c.name)))
            best.accepted = True
            recipe = proposals[best.name]
            model, n_train = fitted[best.name]
            models.append(model)
            st.decision = (f"Accepted {best.name}: gentle-config failure rate on validation "
                           f"{base_rate:.1%} -> {best.gate_failure_rate:.1%}.")
        else:
            models.append(model)
            st.decision = (f"No candidate passed the gate (needs >= {margin:.0%} fewer failures and at "
                           f"most {clean_tol:.1%} clean-accuracy loss on validation); model unchanged.")
        log("  " + st.decision)

    run = FixRun(stats, tau, budget, strategy.name, models=models, history_per_round=histories)
    if reference_evals:
        n_random, n_refine = reference_evals
        for attr, mdl in (("reference_before", models[0]), ("reference_after", models[-1])):
            ev = Evaluator(mdl, splits.X_te, splits.y_te, tau=tau, metric=metric, seed=seed)
            setattr(run, attr, reference_search(ev, space, n_random, n_refine, seed=2024))
        log(f"reference breakpoint on test: {run.reference_before.min_cost:.3f} -> "
            f"{run.reference_after.min_cost:.3f}")
    return run


def fixrun_to_dict(run: FixRun) -> dict:
    def ref(x):
        return None if x is None else dict(min_cost=x.min_cost, fail_fraction=x.fail_fraction,
                                           n_evals=x.n_evals)
    return dict(tau=run.tau, budget=run.budget, strategy=run.strategy,
                rounds=[asdict(s) for s in run.rounds],
                reference_before=ref(run.reference_before), reference_after=ref(run.reference_after))
