"""The search space BreakPoint explores: what to perturb, how hard, and where."""
from __future__ import annotations

import itertools
import zlib
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from .perturbations import FAMILIES

TARGETS = ("random", "important", "unimportant")


@dataclass(frozen=True)
class Config:
    """One stress test.

    family    which perturbation
    severity  in [0, 1], family-specific scale (see perturbations.py)
    coverage  fraction of features touched, in (0, 1]
    target    which features: random, the model's most important, or least important
    seed      makes the stochastic part of the perturbation reproducible
    """

    family: str
    severity: float
    coverage: float
    target: str
    seed: int = 0

    @staticmethod
    def make(family: str, severity: float, coverage: float, target: str, salt: int = 0) -> "Config":
        severity = float(round(severity, 4))
        coverage = float(round(coverage, 4))
        raw = f"{family}|{severity}|{coverage}|{target}|{salt}".encode()
        return Config(family, severity, coverage, target, zlib.crc32(raw) & 0x7FFFFFFF)

    @property
    def cost(self) -> float:
        """Normalized perturbation budget: severity x coverage, in [0, 1].

        Lower cost = a gentler perturbation. A failure at low cost is a weak spot.
        Costs are comparable *within* a family; across families the severity scale is
        a design choice, not a physical equivalence (see README, Limitations).
        """
        return self.severity * self.coverage

    def with_salt(self, salt: int) -> "Config":
        return Config.make(self.family, self.severity, self.coverage, self.target, salt)


class SearchSpace:
    def __init__(self, families: Sequence[str] | None = None,
                 targets: Sequence[str] = TARGETS,
                 sev_range: Tuple[float, float] = (0.02, 1.0),
                 cov_range: Tuple[float, float] = (0.1, 1.0)):
        self.families = list(families) if families else list(FAMILIES)
        self.targets = list(targets)
        self.sev_range = sev_range
        self.cov_range = cov_range
        for f in self.families:
            if f not in FAMILIES:
                raise KeyError(f"unknown family {f!r}")

    @property
    def dim(self) -> int:
        return len(self.families) + len(self.targets) + 2

    def arms(self) -> List[Tuple[str, str]]:
        return list(itertools.product(self.families, self.targets))

    # -- sampling ---------------------------------------------------------
    def sample(self, rng: np.random.Generator) -> Config:
        return Config.make(
            str(rng.choice(self.families)),
            rng.uniform(*self.sev_range),
            rng.uniform(*self.cov_range),
            str(rng.choice(self.targets)),
        )

    def sample_arm(self, arm: Tuple[str, str], rng: np.random.Generator) -> Config:
        return Config.make(arm[0], rng.uniform(*self.sev_range), rng.uniform(*self.cov_range), arm[1])

    def mutate(self, cfg: Config, rng: np.random.Generator, scale: float = 0.12,
               keep_discrete: bool = False) -> Config:
        sev = float(np.clip(cfg.severity + rng.normal(0, scale), *self.sev_range))
        cov = float(np.clip(cfg.coverage + rng.normal(0, scale), *self.cov_range))
        fam, tgt = cfg.family, cfg.target
        if not keep_discrete:
            if rng.random() < 0.15:
                fam = str(rng.choice(self.families))
            if rng.random() < 0.15:
                tgt = str(rng.choice(self.targets))
        return Config.make(fam, sev, cov, tgt)

    # -- encoding for the surrogate --------------------------------------
    def encode(self, cfg: Config) -> np.ndarray:
        v = np.zeros(self.dim)
        v[self.families.index(cfg.family)] = 1.0
        v[len(self.families) + self.targets.index(cfg.target)] = 1.0
        v[-2] = cfg.severity
        v[-1] = cfg.coverage
        return v

    def encode_many(self, cfgs: Sequence[Config]) -> np.ndarray:
        return np.vstack([self.encode(c) for c in cfgs])
