"""Evaluate a fitted model under a stress-test config."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score

from .perturbations import apply_perturbation
from .space import Config

_SCORERS = {
    "accuracy": (lambda y, p: accuracy_score(y, p), "accuracy"),
    "f1_macro": (lambda y, p: f1_score(y, p, average="macro"), "f1_macro"),
}


@dataclass(frozen=True)
class Result:
    config: Config
    score: float      # metric under perturbation
    drop: float       # clean score - perturbed score
    failed: bool      # drop >= tau
    index: int        # order in which it was evaluated (1-based)


class Evaluator:
    """Wraps (fitted model, dataset) and scores stress-test configs.

    * Deterministic: the same Config always gives the same Result.
    * Cached: re-evaluating a Config is free and does not count against a budget.
    * `n_evals` / `history` count unique evaluations, which is what budgets refer to.
    """

    def __init__(self, model, X: np.ndarray, y: np.ndarray, tau: float = 0.10,
                 metric: str = "accuracy", seed: int = 0, importance_repeats: int = 5,
                 _share: "Evaluator | None" = None):
        if metric not in _SCORERS:
            raise KeyError(f"metric must be one of {sorted(_SCORERS)}")
        self.model, self.X, self.y = model, np.asarray(X, dtype=float), np.asarray(y)
        self.tau, self.metric, self.seed = tau, metric, seed
        self._score_fn, scoring = _SCORERS[metric]
        if _share is not None:
            self.clean_score, self.rank = _share.clean_score, _share.rank
        else:
            self.clean_score = self._score_fn(self.y, self._predict(self.X))
            imp = permutation_importance(self.model, self.X, self.y, scoring=scoring,
                                         n_repeats=importance_repeats, random_state=seed, n_jobs=1)
            # most important first; stable so ties are deterministic
            self.rank = np.argsort(-imp.importances_mean, kind="stable")
        self.history: List[Result] = []
        self._cache: Dict[Config, Result] = {}

    def _predict(self, X):
        return np.asarray(self.model.predict(X)).ravel()   # CatBoost returns (n, 1)

    # -- bookkeeping ------------------------------------------------------
    @property
    def n_evals(self) -> int:
        return len(self.history)

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    def fork(self) -> "Evaluator":
        """Fresh history/cache, same model, data and importance ranking."""
        return Evaluator(self.model, self.X, self.y, self.tau, self.metric, self.seed, _share=self)

    def seen(self, cfg: Config) -> bool:
        return cfg in self._cache

    # -- perturbation -----------------------------------------------------
    def columns(self, cfg: Config) -> np.ndarray:
        d = self.n_features
        k = max(1, int(round(cfg.coverage * d)))
        if cfg.target == "important":
            return self.rank[:k].copy()
        if cfg.target == "unimportant":
            return self.rank[::-1][:k].copy()
        return np.random.default_rng(cfg.seed).choice(d, size=k, replace=False)

    def perturb(self, X: np.ndarray, cfg: Config) -> np.ndarray:
        """Apply cfg to any feature matrix (used for evaluation and for augmentation)."""
        rng = np.random.default_rng([cfg.seed, 1])
        return apply_perturbation(X, cfg.family, cfg.severity, self.columns(cfg), rng)

    # -- evaluation -------------------------------------------------------
    def evaluate(self, cfg: Config) -> Result:
        hit = self._cache.get(cfg)
        if hit is not None:
            return hit
        pred = self._predict(self.perturb(self.X, cfg))
        score = float(self._score_fn(self.y, pred))
        drop = float(self.clean_score - score)
        res = Result(cfg, score, drop, drop >= self.tau, len(self.history) + 1)
        self._cache[cfg] = res
        self.history.append(res)
        return res

    def failures(self) -> List[Result]:
        return sorted((r for r in self.history if r.failed), key=lambda r: r.config.cost)
