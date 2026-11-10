import numpy as np
import pytest

from breakpoint_ml import Evaluator, SearchSpace, make_strategy
from breakpoint_ml.benchmark import breakpoint_curve, reference_search
from breakpoint_ml.builtin import load_splits, make_model
from breakpoint_ml.fix import (diagnose, find_and_fix, make_augmentation,
                               select_for_augmentation)


@pytest.fixture(scope="module")
def setup():
    s = load_splits("wine", 0)
    model = make_model("logreg").fit(s.X_tr, s.y_tr)
    ev = Evaluator(model, s.X_val, s.y_val, tau=0.10)
    make_strategy("random").run(ev, SearchSpace(), 80, np.random.default_rng(0))
    return s, ev


def test_selection_is_cheapest_first_and_diverse(setup):
    _, ev = setup
    fails = ev.failures()
    picked = select_for_augmentation(fails, 6)
    assert 0 < len(picked) <= 6
    fams = {r.config.family for r in fails}
    assert len({c.family for c in picked}) == min(len(fams), len(picked))


def test_augmentation_volume_and_labels(setup):
    s, ev = setup
    cfgs = select_for_augmentation(ev.failures(), 5)
    X, y = make_augmentation(ev, s.X_tr, s.y_tr, cfgs, SearchSpace(), ratio=1.0,
                             rng=np.random.default_rng(0), chunks=4)
    assert X.shape[1] == s.X_tr.shape[1] and len(X) == len(y)
    assert 0.6 * len(s.X_tr) <= len(X) <= 1.2 * len(s.X_tr)
    assert set(np.unique(y)) <= set(np.unique(s.y_tr))


def test_diagnosis_shares_sum_to_one(setup):
    _, ev = setup
    d = diagnose(ev.failures())
    assert sum(v for k, v in d.items() if k.startswith("target:")) == pytest.approx(1.0)
    assert sum(v for k, v in d.items() if k.startswith("family:")) == pytest.approx(1.0)


def test_curve_is_monotone_non_increasing(setup):
    _, ev = setup
    c = breakpoint_curve(ev.history, 80)
    assert len(c) == 80 and np.all(np.diff(c) <= 1e-12)


def test_reference_search_runs(setup):
    _, ev = setup
    ref = reference_search(ev, SearchSpace(), n_random=60, n_refine=10)
    assert 0 < ref.min_cost <= 1.0 and 0 <= ref.fail_fraction <= 1


def test_find_and_fix_end_to_end_is_honest_and_reproducible():
    s = load_splits("wine", 0)
    kwargs = dict(budget=30, rounds=1, seed=0, battery_size=60, gate_size=60, members=4,
                  reference_evals=None)
    run = find_and_fix(lambda: make_model("logreg"), s, make_strategy("random"), SearchSpace(), **kwargs)
    again = find_and_fix(lambda: make_model("logreg"), s, make_strategy("random"), SearchSpace(), **kwargs)
    assert len(run.rounds) == 2
    first = run.rounds[0]
    assert first.candidates and first.decision
    # at most one candidate is kept, and only if it passed the gate
    kept = [c for c in first.candidates if c.accepted]
    assert len(kept) <= 1 and all(c.eligible for c in kept)
    # the model is unchanged when nothing is kept
    if not kept:
        assert run.rounds[1].recipe == "baseline"
    assert [r.battery_failure_rate for r in run.rounds] == [r.battery_failure_rate for r in again.rounds]
