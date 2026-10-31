"""Perturbation families for stress-testing tabular ML models.

Every family maps a *severity* s in [0, 1] to a concrete perturbation and acts on a
chosen set of feature columns. Data is assumed to be standardized (zero mean, unit
variance per feature), so magnitudes are expressed in "standard deviations".

s = 0 is the identity for every family, which the tests check.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Sequence

import numpy as np

PerturbFn = Callable[[np.ndarray, Sequence[int], float, np.random.Generator], np.ndarray]


def _gaussian_noise(X, cols, s, rng):
    out = X.copy()
    out[:, cols] += rng.normal(0.0, 3.0 * s, size=(X.shape[0], len(cols)))
    return out


def _feature_dropout(X, cols, s, rng):
    """Missing values imputed with the training mean (0 in standardized space)."""
    out = X.copy()
    block = out[:, cols]
    block[rng.random(block.shape) < s] = 0.0
    out[:, cols] = block
    return out


def _mean_shift(X, cols, s, rng):
    """Covariate shift: every selected feature moves by +/- 3s standard deviations."""
    out = X.copy()
    signs = rng.choice([-1.0, 1.0], size=len(cols))
    out[:, cols] += 3.0 * s * signs
    return out


def _scale_distortion(X, cols, s, rng):
    """Unit/calibration drift: half the selected features are amplified, half shrunk."""
    out = X.copy()
    factor = 1.0 + 3.0 * s
    amplify = rng.random(len(cols)) < 0.5
    out[:, cols] *= np.where(amplify, factor, 1.0 / factor)
    return out


def _outliers(X, cols, s, rng):
    """20% of rows get large +/- spikes in the selected features."""
    out = X.copy()
    n = X.shape[0]
    rows = rng.random(n) < 0.2
    signs = rng.choice([-1.0, 1.0], size=(n, len(cols)))
    block = out[:, cols]
    block[rows] += signs[rows] * 8.0 * s
    out[:, cols] = block
    return out


def _quantization(X, cols, s, rng):
    """Sensor coarsening: values are rounded to a grid of step 4s."""
    out = X.copy()
    step = 4.0 * s
    if step < 1e-6:
        return out
    out[:, cols] = np.round(out[:, cols] / step) * step
    return out


def _correlated_noise(X, cols, s, rng):
    """One shared noise draw per row added to all selected features (common-mode drift)."""
    out = X.copy()
    z = rng.normal(0.0, 1.0, size=(X.shape[0], 1))
    out[:, cols] += 3.0 * s * z
    return out


def _feature_shuffle(X, cols, s, rng):
    """Break the link between the selected feature block and the rest of the row.

    A fraction s of rows have their selected-feature block swapped with another
    affected row's block. Marginals are preserved; cross-feature structure is not.
    """
    out = X.copy()
    idx = np.flatnonzero(rng.random(X.shape[0]) < s)
    if idx.size < 2:
        return out
    perm = rng.permutation(idx)
    block = out[:, cols]
    block[idx] = block[perm]
    out[:, cols] = block
    return out


@dataclass(frozen=True)
class FamilySpec:
    name: str
    fn: PerturbFn
    describe: Callable[[float], str]


FAMILIES: Dict[str, FamilySpec] = {
    spec.name: spec
    for spec in [
        FamilySpec("gaussian_noise", _gaussian_noise,
                   lambda s: f"independent Gaussian noise (sigma = {3 * s:.2f} std)"),
        FamilySpec("feature_dropout", _feature_dropout,
                   lambda s: f"missing values (mean-imputed) in {100 * s:.0f}% of entries"),
        FamilySpec("mean_shift", _mean_shift,
                   lambda s: f"covariate shift of +/-{3 * s:.2f} std"),
        FamilySpec("scale_distortion", _scale_distortion,
                   lambda s: f"unit drift (x{1 + 3 * s:.2f} or /{1 + 3 * s:.2f})"),
        FamilySpec("outliers", _outliers,
                   lambda s: f"+/-{8 * s:.1f} std spikes in 20% of rows"),
        FamilySpec("quantization", _quantization,
                   lambda s: f"coarse rounding to steps of {4 * s:.2f} std"),
        FamilySpec("correlated_noise", _correlated_noise,
                   lambda s: f"shared common-mode noise (sigma = {3 * s:.2f} std)"),
        FamilySpec("feature_shuffle", _feature_shuffle,
                   lambda s: f"feature block swapped between {100 * s:.0f}% of rows"),
    ]
}


def apply_perturbation(X: np.ndarray, family: str, severity: float,
                       cols: Sequence[int], rng: np.random.Generator) -> np.ndarray:
    """Return a perturbed copy of X. The input is never modified."""
    if family not in FAMILIES:
        raise KeyError(f"unknown perturbation family {family!r}; choose from {sorted(FAMILIES)}")
    if not 0.0 <= severity <= 1.0:
        raise ValueError(f"severity must be in [0, 1], got {severity}")
    cols = list(cols)
    if not cols:
        return X.copy()
    return FAMILIES[family].fn(np.asarray(X, dtype=float), cols, float(severity), rng)


def describe(family: str, severity: float) -> str:
    return FAMILIES[family].describe(severity)
