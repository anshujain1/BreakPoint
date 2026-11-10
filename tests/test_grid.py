import json
import os

import numpy as np
import pandas as pd
import pytest

from breakpoint_ml import Evaluator
from breakpoint_ml.aggregate import boot_ci, fix_stats, search_stats, write_summary
from breakpoint_ml.builtin import MODELS, dataset_names, load_csv_splits, load_splits, make_model
from breakpoint_ml.grid import cell_path, load_cells, run_cell, run_grid


@pytest.mark.parametrize("name", ["synthetic_large", "nonlinear", "multiclass"])
def test_new_datasets_load_with_clean_splits(name):
    s = load_splits(name, 0)
    assert len(s.y_tr) > 1000 and s.X_tr.shape[1] == s.X_val.shape[1] == s.X_te.shape[1]
    np.testing.assert_allclose(s.X_tr.mean(0), 0, atol=1e-8)       # scaler fit on train only
    assert set(np.unique(s.y_val)) <= set(np.unique(s.y_tr))
    # different seeds give different splits of the same fixed data
    other = load_splits(name, 1)
    assert not np.array_equal(other.X_val[:20], s.X_val[:20])
    assert len(other.y_tr) == len(s.y_tr)


@pytest.mark.parametrize("model", ["xgboost", "lightgbm", "catboost"])
def test_boosting_models_work_with_evaluator(model):
    pytest.importorskip({"xgboost": "xgboost", "lightgbm": "lightgbm", "catboost": "catboost"}[model])
    s = load_splits("multiclass", 0)
    m = make_model(model, 0).fit(s.X_tr[:800], s.y_tr[:800])
    ev = Evaluator(m, s.X_val[:200], s.y_val[:200], importance_repeats=2)
    assert ev.rank.shape == (s.X_tr.shape[1],)
    assert ev.clean_score > 0.4                       # 5 classes: chance is 0.2
    from breakpoint_ml import Config
    r = ev.evaluate(Config.make("gaussian_noise", 1.0, 1.0, "random"))
    assert r.score < ev.clean_score                   # predictions are 1-D, scoring works (CatBoost returns (n, 1))


def test_csv_loader_handles_categoricals_missing_and_string_target(tmp_path):
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n),
                       "color": rng.choice(["red", "green", "blue"], n),
                       "label": rng.choice(["yes", "no"], n)})
    df.loc[0, "a"] = np.nan
    path = tmp_path / "d.csv"
    df.to_csv(path, index=False)
    s = load_csv_splits(str(path), "label", seed=0)
    assert s.X_tr.shape[1] == 2 + 3                   # two numerics + three one-hot columns
    assert len(s.y_tr) + len(s.y_val) + len(s.y_te) == n - 1   # the row with a NaN was dropped
    assert set(np.unique(s.y_tr)) == {0, 1}
    with pytest.raises(KeyError):
        load_csv_splits(str(path), "nope")


def test_all_registered_models_construct():
    for name in MODELS:
        try:
            make_model(name, 0)
        except ImportError:
            pytest.skip(f"{name} not installed")


FAST = dict(budget=14, ref_evals=(25, 5), rounds=1,
            fix_kwargs=dict(battery_size=40, gate_size=40, members=3))


def test_run_cell_returns_complete_record():
    c = run_cell("wine", "logreg", 0, **FAST)
    assert c["ok"] and c["dataset"] == "wine" and c["seed"] == 0
    assert set(c["search"]) == {"random", "bandit", "surrogate"}
    for s in c["search"].values():
        assert len(s["curve"]) == 14 and all(b <= a + 1e-12 for a, b in zip(s["curve"], s["curve"][1:]))
    assert len(c["fix"]["rounds"]) == 2
    json.dumps(c)                                       # must be serializable


def test_run_cell_is_reproducible():
    a = run_cell("wine", "logreg", 3, **FAST)
    b = run_cell("wine", "logreg", 3, **FAST)
    assert [a["search"][k]["final_cost"] for k in a["search"]] == [b["search"][k]["final_cost"] for k in b["search"]]
    assert a["fix"]["rounds"][0]["battery_failure_rate"] == b["fix"]["rounds"][0]["battery_failure_rate"]


def test_grid_is_resumable_and_records_failures(tmp_path):
    out = str(tmp_path)
    run_grid(["logreg"], ["wine"], [0], out, workers=1, log=lambda m: None, **FAST)
    assert os.path.exists(cell_path(out, "wine", "logreg", 0))
    first = os.path.getmtime(cell_path(out, "wine", "logreg", 0))
    logs = []
    run_grid(["logreg"], ["wine"], [0], out, workers=1, log=logs.append, **FAST)     # nothing left to do
    assert os.path.getmtime(cell_path(out, "wine", "logreg", 0)) == first and "1 already done" in logs[0]
    # an impossible cell is recorded as failed, not swallowed
    logs = []
    run_grid(["logreg"], ["no_such_dataset"], [0], out, workers=1, log=logs.append, **FAST)
    bad = [c for c in load_cells(out) if not c["ok"]]
    assert len(bad) == 1 and "KeyError" in bad[0]["error"] and "FAILED" in logs[-1]
    assert len(load_cells(out)) == 2


def _fake_cell(ds, model, seed, rand, band, sur, ref=0.05, fail_frac=0.2, base_low=0.2, final_low=0.1,
               accepted=True):
    def s(cost):
        return dict(final_cost=cost, n_failures=5, first_failure=3, evals_to_near_ref=10, seconds=1.0,
                    curve=[1.0] * 5 + [cost] * 15)
    rnd = lambda recipe, low: dict(round=0, recipe=recipe, battery_low_cost_failure_rate=low,
                                   battery_failure_rate=low * 2, clean_test=0.9,
                                   candidates=[dict(name="targeted_augmentation", gate_failure_rate=0.1),
                                               dict(name="random_augmentation", gate_failure_rate=0.15)])
    return dict(ok=True, dataset=ds, model=model, seed=seed, budget=20, tau=0.1,
                ref=dict(min_cost=ref, fail_fraction=fail_frac, n_evals=100),
                search=dict(random=s(rand), bandit=s(band), surrogate=s(sur)),
                fix=dict(rounds=[rnd("baseline", base_low), rnd("targeted_augmentation(r1)" if accepted else "baseline",
                                                                final_low if accepted else base_low)]))


def test_search_stats_and_fix_stats_on_known_data():
    cells = [_fake_cell("d", "m", i, 0.2, 0.2, 0.1) for i in range(6)]
    cells.append(_fake_cell("d", "m", 9, 0.2, 0.2, 0.1, fail_frac=0.0))              # excluded: nothing to find
    cells.append(_fake_cell("d", "m", 10, 0.1, 0.1, 0.2, accepted=False))            # a loss for the surrogate
    S = search_stats(cells)
    assert S["n_valid"] == 7 and S["n_no_failures"] == 1
    v = S["vs"]["random"]
    assert (v["wins"], v["losses"], v["ties"]) == (6, 1, 0)
    assert v["lo"] <= v["gm_ratio"] <= v["hi"] and v["gm_ratio"] < 1
    F = fix_stats(cells)
    assert F["n_accepted"] == 7 and F["accepted_better"] == 7
    assert F["accepted_d_low"][0] == pytest.approx(-0.1)
    assert F["targeted_vs_random"]["wins"] == F["targeted_vs_random"]["n"]


def test_boot_ci_contains_mean_and_handles_empty():
    m, lo, hi = boot_ci([1.0, 2.0, 3.0, 4.0])
    assert lo <= m <= hi and m == pytest.approx(2.5)
    assert np.isnan(boot_ci([])[0])


def test_write_summary_drops_incomplete_seeds(tmp_path):
    cells_dir = tmp_path / "cells"
    cells_dir.mkdir()
    for sd in (0, 1):
        for m in ("a", "b"):
            c = _fake_cell("d", m, sd, 0.2, 0.2, 0.1)
            c.update(n_train=10, n_val=5, n_test=5)
            (cells_dir / f"d__{m}__{sd}.json").write_text(json.dumps(c))
    c = _fake_cell("d", "a", 2, 0.2, 0.2, 0.1)                                       # seed 2: only 1 of 2 models
    (cells_dir / "d__a__2.json").write_text(json.dumps(c))
    path = write_summary(str(tmp_path))
    text = open(path).read()
    assert "4 cells" in text and "Seeds [2] were incomplete" in text
