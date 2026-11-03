"""Search strategies: given an evaluation budget, find cheap ways to break the model.

All strategies share one objective, the *utility* of a result:

    utility = min(drop / tau, 1) * (1 - cost)

It is 1.0 for a failure that costs nothing, and shrinks both when the damage is below
the failure threshold and when the perturbation had to be large to cause it. Maximizing
utility therefore means: find failures, and find the gentlest ones.
"""
from __future__ import annotations

import warnings
from typing import List

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

from .evaluator import Evaluator, Result
from .space import SearchSpace


def utility(res: Result, tau: float) -> float:
    return min(max(res.drop, 0.0) / tau, 1.0) * (1.0 - res.config.cost)


class Strategy:
    name = "base"

    def run(self, ev: Evaluator, space: SearchSpace, budget: int,
            rng: np.random.Generator) -> List[Result]:
        raise NotImplementedError


class RandomSearch(Strategy):
    """Uniform sampling. The baseline every smarter strategy has to beat."""
    name = "random"

    def run(self, ev, space, budget, rng):
        while ev.n_evals < budget:
            ev.evaluate(space.sample(rng))
        return list(ev.history)


class BanditSearch(Strategy):
    """UCB1 over (family, target) arms, with local refinement inside the best region of an arm.

    This is the 'before' design: it learns which kinds of perturbation hurt, but has no
    model of how severity and coverage interact, so it cannot extrapolate to unseen configs.
    """
    name = "bandit"

    def __init__(self, c: float = 0.5, local_prob: float = 0.5):
        self.c, self.local_prob = c, local_prob

    def run(self, ev, space, budget, rng):
        arms = space.arms()
        n = {a: 0 for a in arms}
        total = {a: 0.0 for a in arms}
        best = {}  # arm -> (utility, config)
        t = 0
        while ev.n_evals < budget:
            t += 1
            untried = [a for a in arms if n[a] == 0]
            if untried:
                arm = untried[int(rng.integers(len(untried)))]
            else:
                ucb = [total[a] / n[a] + self.c * np.sqrt(2.0 * np.log(t) / n[a]) for a in arms]
                arm = arms[int(np.argmax(ucb))]
            if arm in best and rng.random() < self.local_prob:
                cfg = space.mutate(best[arm][1], rng, scale=0.1, keep_discrete=True)
            else:
                cfg = space.sample_arm(arm, rng)
            before = ev.n_evals
            res = ev.evaluate(cfg)
            if ev.n_evals == before:   # duplicate after rounding; nothing learned, try again
                continue
            u = utility(res, ev.tau)
            n[arm] += 1
            total[arm] += u
            if arm not in best or u > best[arm][0]:
                best[arm] = (u, cfg)
        return list(ev.history)


class SurrogateSearch(Strategy):
    """Gaussian-process surrogate over the config space with a utility-UCB acquisition.

    1. Spend a small random warm-up budget.
    2. Fit a GP (ARD Matern kernel) mapping encoded config -> accuracy drop.
    3. Score a large pool of candidate configs *without running the model*, using the
       optimistic drop (mean + kappa * std) pushed through the same utility as above.
    4. Evaluate the best candidate for real, add it to the data, repeat.

    Because the surrogate generalizes across severity, coverage, family and target, it
    can propose configs it has never tried and concentrate near the failure boundary.
    A small epsilon of pure random picks guards against a surrogate stuck in a wrong belief.
    """
    name = "surrogate"

    def __init__(self, warmup_frac: float = 0.15, min_warmup: int = 12, kappa: float = 1.0,
                 n_random: int = 1200, n_local: int = 60, epsilon: float = 0.08,
                 refit_every: int = 4):
        self.warmup_frac, self.min_warmup, self.kappa = warmup_frac, min_warmup, kappa
        self.n_random, self.n_local, self.epsilon = n_random, n_local, epsilon
        self.refit_every = refit_every

    def _gp(self, dim, rng, kernel=None, optimize=True):
        if kernel is None:
            kernel = (C(1.0, (1e-2, 1e2))
                      * Matern(length_scale=np.ones(dim), length_scale_bounds=(0.05, 20.0), nu=2.5)
                      + WhiteKernel(1e-2, (1e-6, 1e-1)))
        return GaussianProcessRegressor(
            kernel=kernel, normalize_y=True,
            optimizer="fmin_l_bfgs_b" if optimize else None,
            n_restarts_optimizer=1 if optimize else 0,
            random_state=int(rng.integers(2**31 - 1)))

    def run(self, ev, space, budget, rng):
        warm = min(budget, max(self.min_warmup, int(self.warmup_frac * budget)))
        while ev.n_evals < warm:
            ev.evaluate(space.sample(rng))

        gp, step = None, 0
        while ev.n_evals < budget:
            if rng.random() < self.epsilon:
                ev.evaluate(space.sample(rng))
                continue

            hist = ev.history
            X = space.encode_many([r.config for r in hist])
            y = np.array([r.drop for r in hist])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                warnings.simplefilter("ignore", UserWarning)
                refit = gp is None or step % self.refit_every == 0
                gp = self._gp(space.dim, rng, None if refit else gp.kernel_, optimize=refit).fit(X, y)
                step += 1

                top = sorted(hist, key=lambda r: utility(r, ev.tau), reverse=True)[:5]
                pool = [space.sample(rng) for _ in range(self.n_random)]
                for r in top:
                    for scale in (0.06, 0.2):
                        pool += [space.mutate(r.config, rng, scale=scale)
                                 for _ in range(self.n_local // 2)]
                pool = [c for c in pool if not ev.seen(c)]
                if not pool:
                    ev.evaluate(space.sample(rng))
                    continue
                mu, sd = gp.predict(space.encode_many(pool), return_std=True)

            optimistic = np.clip(mu + self.kappa * sd, 0.0, None)
            cost = np.array([c.cost for c in pool])
            acq = np.minimum(optimistic / ev.tau, 1.0) * (1.0 - cost)
            ev.evaluate(pool[int(np.argmax(acq))])
        return list(ev.history)


STRATEGIES = {s.name: s for s in (RandomSearch, BanditSearch, SurrogateSearch)}


def make_strategy(name: str) -> Strategy:
    if name not in STRATEGIES:
        raise KeyError(f"unknown strategy {name!r}; choose from {sorted(STRATEGIES)}")
    return STRATEGIES[name]()
