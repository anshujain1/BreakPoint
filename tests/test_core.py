import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from breakpoint_ml import Config, Evaluator, SearchSpace, make_strategy
from breakpoint_ml.builtin import load_splits
from breakpoint_ml.perturbations import FAMILIES, apply_perturbation


@pytest.fixture(scope="module")
def setup():
    s = load_splits("wine", 0)
    model = LogisticRegression(max_iter=1000).fit(s.X_tr, s.y_tr)
    return s, model


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_zero_severity_is_identity(family):
    X = np.random.default_rng(0).normal(size=(50, 6))
    out = apply_perturbation(X, family, 0.0, [0, 2, 4], np.random.default_rng(1))
    np.testing.assert_allclose(out, X)


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_only_selected_columns_change_and_input_untouched(family):
    X = np.random.default_rng(0).normal(size=(60, 6))
    X0 = X.copy()
    out = apply_perturbation(X, family, 0.8, [1, 3], np.random.default_rng(2))
    np.testing.assert_array_equal(X, X0)
    untouched = [0, 2, 4, 5]
    np.testing.assert_array_equal(out[:, untouched], X0[:, untouched])
    assert not np.allclose(out[:, [1, 3]], X0[:, [1, 3]])


def test_deterministic_given_rng():
    X = np.random.default_rng(0).normal(size=(40, 5))
    a = apply_perturbation(X, "gaussian_noise", 0.5, [0, 1], np.random.default_rng(9))
    b = apply_perturbation(X, "gaussian_noise", 0.5, [0, 1], np.random.default_rng(9))
    np.testing.assert_array_equal(a, b)


def test_bad_inputs_rejected():
    X = np.zeros((3, 2))
    with pytest.raises(KeyError):
        apply_perturbation(X, "nope", 0.5, [0], np.random.default_rng(0))
    with pytest.raises(ValueError):
        apply_perturbation(X, "gaussian_noise", 1.5, [0], np.random.default_rng(0))


def test_config_seed_is_stable_and_cost_is_product():
    a = Config.make("mean_shift", 0.5, 0.4, "random")
    b = Config.make("mean_shift", 0.5, 0.4, "random")
    assert a == b and a.seed == b.seed
    assert a.cost == pytest.approx(0.2)
    assert a.with_salt(3) != a


def test_space_sampling_and_encoding():
    space = SearchSpace()
    rng = np.random.default_rng(0)
    for _ in range(100):
        c = space.sample(rng)
        assert space.sev_range[0] <= c.severity <= space.sev_range[1]
        assert space.cov_range[0] <= c.coverage <= space.cov_range[1]
        v = space.encode(c)
        assert v.shape == (space.dim,)
        assert v[: len(space.families)].sum() == 1 and v[len(space.families):-2].sum() == 1
        m = space.mutate(c, rng)
        assert space.sev_range[0] <= m.severity <= space.sev_range[1]


def test_evaluator_caches_counts_and_is_deterministic(setup):
    s, model = setup
    ev = Evaluator(model, s.X_val, s.y_val, tau=0.05)
    cfg = Config.make("gaussian_noise", 0.9, 1.0, "random")
    r1 = ev.evaluate(cfg)
    r2 = ev.evaluate(cfg)
    assert r1 is r2 and ev.n_evals == 1
    fork = ev.fork()
    assert fork.n_evals == 0 and fork.evaluate(cfg).score == r1.score
    assert r1.drop == pytest.approx(ev.clean_score - r1.score)
    assert r1.failed == (r1.drop >= 0.05)


def test_columns_follow_importance(setup):
    s, model = setup
    ev = Evaluator(model, s.X_val, s.y_val)
    k = 3
    cov = k / ev.n_features
    top = set(ev.columns(Config.make("gaussian_noise", 0.5, cov, "important")))
    bottom = set(ev.columns(Config.make("gaussian_noise", 0.5, cov, "unimportant")))
    assert top == set(ev.rank[:k]) and bottom == set(ev.rank[::-1][:k])
    assert len(set(ev.columns(Config.make("gaussian_noise", 0.5, cov, "random")))) == k


@pytest.mark.parametrize("name", ["random", "bandit", "surrogate"])
def test_strategies_respect_budget_and_never_repeat(setup, name):
    s, model = setup
    ev = Evaluator(model, s.X_val, s.y_val, tau=0.10)
    hist = make_strategy(name).run(ev, SearchSpace(), 30, np.random.default_rng(0))
    assert len(hist) == 30 == ev.n_evals
    assert len({r.config for r in hist}) == 30


def test_strategies_find_a_failure_on_a_fragile_model(setup):
    s, model = setup
    ev = Evaluator(model, s.X_val, s.y_val, tau=0.10)
    make_strategy("surrogate").run(ev, SearchSpace(), 40, np.random.default_rng(1))
    assert ev.failures(), "expected at least one failure within 40 evaluations"
    costs = [r.config.cost for r in ev.failures()]
    assert costs == sorted(costs)
