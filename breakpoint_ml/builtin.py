"""Built-in datasets and models so BreakPoint runs offline with zero setup."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_digits, load_wine, make_classification
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass
class Splits:
    """train: fit the model. val: where failures are searched for. test: untouched until reporting."""
    X_tr: np.ndarray
    y_tr: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_te: np.ndarray
    y_te: np.ndarray
    feature_names: list


def _synthetic():
    X, y = make_classification(n_samples=1500, n_features=20, n_informative=8, n_redundant=4,
                               n_classes=2, class_sep=1.2, random_state=7)
    return X, y, [f"f{i}" for i in range(X.shape[1])]


def _synthetic_large():
    """6,000 rows, 40 features (12 informative), 3% label noise: bigger and noisier than the toys."""
    X, y = make_classification(n_samples=6000, n_features=40, n_informative=12, n_redundant=8,
                               n_classes=2, flip_y=0.03, class_sep=1.0, random_state=11)
    return X, y, [f"f{i}" for i in range(X.shape[1])]


def _nonlinear():
    """Friedman-style interactions: y = [10 sin(pi x0 x1) + 20 (x2-.5)^2 + 10 x3 + 5 x4 + noise > median].
    Linear models struggle here; tree/boosting/deep models should not."""
    rng = np.random.default_rng(5)
    X = rng.uniform(size=(6000, 20))
    f = (10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2 + 10 * X[:, 3]
         + 5 * X[:, 4] + rng.normal(0, 1.0, size=6000))
    return X, (f > np.median(f)).astype(int), [f"x{i}" for i in range(20)]


def _multiclass():
    """5 classes, 2 clusters per class, 30 features (14 informative)."""
    X, y = make_classification(n_samples=5000, n_features=30, n_informative=14, n_redundant=4,
                               n_classes=5, n_clusters_per_class=2, class_sep=1.0, random_state=3)
    return X, y, [f"f{i}" for i in range(X.shape[1])]


_DATASETS: Dict[str, Callable] = {
    "breast_cancer": lambda: (*load_breast_cancer(return_X_y=True), list(load_breast_cancer().feature_names)),
    "wine": lambda: (*load_wine(return_X_y=True), list(load_wine().feature_names)),
    "digits": lambda: (*load_digits(return_X_y=True), [f"px{i}" for i in range(64)]),
    "synthetic": _synthetic,
    "synthetic_large": _synthetic_large,
    "nonlinear": _nonlinear,
    "multiclass": _multiclass,
}


def _xgboost(seed=0):
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.8,
                         colsample_bytree=0.8, tree_method="hist", n_jobs=1, random_state=seed,
                         verbosity=0)


def _lightgbm(seed=0):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=200, num_leaves=31, learning_rate=0.1, subsample=0.8,
                          subsample_freq=1, colsample_bytree=0.8, n_jobs=1, random_state=seed,
                          verbose=-1)


def _catboost(seed=0):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(iterations=150, depth=6, learning_rate=0.1, thread_count=1,
                              random_seed=seed, verbose=0, allow_writing_files=False)


MODELS: Dict[str, Callable] = {
    "xgboost": _xgboost,
    "lightgbm": _lightgbm,
    "catboost": _catboost,
    "logreg": lambda seed=0: LogisticRegression(max_iter=2000, random_state=seed),
    "rf": lambda seed=0: RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1),
    "gb": lambda seed=0: HistGradientBoostingClassifier(random_state=seed),
    "svm": lambda seed=0: SVC(kernel="rbf", random_state=seed),
    "mlp": lambda seed=0: MLPClassifier(hidden_layer_sizes=(64,), max_iter=600, random_state=seed),
    "knn": lambda seed=0: KNeighborsClassifier(n_neighbors=5),
}


def dataset_names():
    return sorted(_DATASETS)


def _split(X, y, feats, seed) -> Splits:
    X_tr, X_rest, y_tr, y_rest = train_test_split(X, y, test_size=0.5, stratify=y, random_state=seed)
    X_val, X_te, y_val, y_te = train_test_split(X_rest, y_rest, test_size=0.5, stratify=y_rest,
                                                random_state=seed)
    scaler = StandardScaler().fit(X_tr)
    return Splits(scaler.transform(X_tr), y_tr, scaler.transform(X_val), y_val,
                  scaler.transform(X_te), y_te, feats)


def load_csv_splits(path: str, target: str, seed: int = 0, max_rows: int | None = None) -> Splits:
    """Load your own tabular data. Categorical columns are one-hot encoded, rows with missing
    values are dropped, the target is label-encoded. Same 50/25/25 stratified split as the built-ins."""
    df = pd.read_csv(path)
    if target not in df.columns:
        raise KeyError(f"target column {target!r} not in {list(df.columns)[:10]}...")
    df = df.dropna()
    if max_rows and len(df) > max_rows:
        df = df.sample(max_rows, random_state=seed)
    y = pd.factorize(df[target])[0]
    X = pd.get_dummies(df.drop(columns=[target]), dtype=float)
    return _split(X.to_numpy(dtype=float), y, list(X.columns), seed)


def load_splits(name: str, seed: int = 0) -> Splits:
    """Stratified 50/25/25 train/val/test split; scaler is fit on train only.
    For synthetic datasets the data itself is fixed; `seed` varies the split (and model init)."""
    if name not in _DATASETS:
        raise KeyError(f"unknown dataset {name!r}; choose from {dataset_names()}")
    X, y, feats = _DATASETS[name]()
    return _split(X, y, feats, seed)


def make_model(name: str, seed: int = 0):
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    return MODELS[name](seed)
