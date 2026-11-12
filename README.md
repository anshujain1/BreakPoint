# BreakPoint

Adaptive stress-testing and find-and-fix for ML classifiers.

BreakPoint answers: *how gently can I perturb my model's inputs and still make it fail, and can I fix it?*
It searches a space of stress tests (what to perturb, how hard, on which features), reports the
cheapest ways the model breaks, then tries fixes and **keeps one only if it is verified to help**.

```
python -m breakpoint_ml demo --out runs/demo                                   # two worked examples + report
python -m breakpoint_ml run --dataset digits --model xgboost --out runs/x      # one dataset/model
python -m breakpoint_ml run --csv my.csv --target label --model lightgbm       # your own data
python -m breakpoint_ml grid --seeds 5 --out runs/grid                         # multi-seed benchmark (resumable)
python -m breakpoint_ml aggregate --out runs/grid                              # rebuild the grid report
pytest                                                                         # 46 tests, ~30 s
```

Requires Python 3.10+ and `pip install -r requirements.txt` (numpy, scipy, pandas, scikit-learn, matplotlib,
xgboost, lightgbm, catboost). Everything runs offline.

**Models:** logreg, rf, gb (sklearn), **xgboost, lightgbm, catboost**, mlp, svm, knn.
**Datasets:** breast_cancer, wine, digits (real, built into scikit-learn) and synthetic, synthetic_large,
nonlinear, multiclass (generated), or **any CSV** via `--csv/--target` (categoricals one-hot encoded, rows with
missing values dropped).

## Web app

```
python -m breakpoint_ml.server      # http://127.0.0.1:8000
```

Upload a trained classifier (`.joblib` / `.pkl`) and a held-out CSV, or let BreakPoint train a built-in model on your CSV.
It finds the cheapest failure, then **Fix it** runs the propose-verify-accept loop. For an uploaded model it retrains a
`sklearn.base.clone` of it on the training CSV you provide, and the accepted fix can be downloaded as a `.joblib`
pipeline that takes your raw features. Local use only: loading a `.joblib` runs code, so only upload models you trust.

## How it works

**Stress tests.** A test is `(family, severity, coverage, target)`:

| | |
|---|---|
| family | gaussian noise, missing values, mean shift, unit/scale drift, outlier spikes, coarse rounding, shared common-mode noise, feature-block shuffle |
| severity | 0-1, scaled per family (e.g. noise sigma = 3 x severity std) |
| coverage | fraction of features touched |
| target | random features, the model's **most** important, or its **least** important (permutation importance) |

`cost = severity x coverage`. A test *fails* when accuracy drops by at least `tau` (default 10 points).
The goal is the **cheapest failure** (the "breakpoint"), not just any failure.

**Search** (`strategies.py`): three strategies, same budget, same objective
`utility = min(drop / tau, 1) x (1 - cost)`:
- `random`: uniform sampling (baseline)
- `bandit`: UCB1 over (family, target) arms with local refinement (the original design)
- `surrogate`: Gaussian-process model of config -> accuracy drop; scores ~1,500 candidate tests per step
  without running the model and evaluates the most promising one. Default.

**Fix loop** (`fix.py`). Each round: search -> diagnose -> train candidate fixes (`targeted_augmentation`,
`random_augmentation` as a blind control, `feature_subspace` ensemble) -> **verify** each on a validation gate
-> keep the best one that passes, otherwise change nothing. Gate: at least 2 points fewer failures among gentle
configs (cost <= 0.25) and at most 1.5 points of clean accuracy lost (never stricter than one validation sample).

**Data discipline.** `train` fits models. `val` drives the search and the accept/reject gate. `test` is only used
for reported before/after numbers.

## Results: 6 models x 6 datasets x 3 seeds = 108 cells (0 failed)

Full tables and charts: `runs/grid/grid_report.md`. Search budget 100 evaluations per strategy per cell. The seed
changes the split, model init and search RNG. (For speed in the grid: validation capped at 800 rows, test at
1,000, one fix round, 5-member subspace ensembles; training always uses the full train split.)

**Finding failures: the surrogate search is clearly better.** Cheapest failure found, surrogate vs baseline
(ratio below 1 = gentler failure found):

| surrogate vs | ratio (geo-mean) | wins / losses over 108 cells |
|---|---|---|
| random | **0.38** (95% CI 0.35-0.42); per seed 0.46 / 0.34 / 0.36 | 106 / 2 |
| bandit | **0.52** (95% CI 0.48-0.57); per seed 0.40 / 0.67 / 0.54 | 100 / 8 |

It wins for every model family (xgboost, lightgbm, catboost, rf, mlp, logreg: 17-18 of 18 cells each) and every
dataset (16-18 of 18). After 100 evaluations it lands within 1.25x of a 750-evaluation reference search in 86 of
108 cells; random search does in 2. It costs about 7 s per search (the GP) versus under 1 s for the others, which
is negligible next to the model evaluations on anything real. It finds fewer failing configs overall (19.7 vs 37.7
for random) because it concentrates on the boundary instead of sampling broadly; if you want a broad sample of
failures, use random.

**Fixing: it works sometimes and is a small effect overall.**
- A fix was accepted in 28 of 108 cells (20 targeted augmentation, 7 random augmentation, 1 feature subspace); in
  the other 80 the gate rejected everything and the model was left alone.
- Among the 28 accepted: 21 reduced gentle-config failures on the untouched test split, 6 made them worse, 1 was
  unchanged. Mean change **-3.6 points** (95% CI -5.3 to -1.7), clean accuracy -0.1 points.
- Across all 108 cells the mean change is **-0.9 points** (95% CI -1.5 to -0.4), from a baseline average of
  20.2% of gentle configs breaking the model to 19.3%. That is real but small.
- It helps mostly the boosted-tree models (lightgbm -2.4 pts, xgboost -2.7 pts averaged over all their cells).
  It does essentially nothing for rf, mlp and logreg.
- Targeted augmentation did **not** clearly beat blind random augmentation on the validation gate: lower in 54
  cells, tied in 11, higher in 43. So I would not claim that "learning from the failures" is the reason fixes
  work; part of the effect is generic noise-robust training.

## Things you should know before quoting any of this

- **Confidence intervals are over cells, not over problems.** Cells share datasets and models, and the synthetic
  datasets are fixed across seeds. Also, the random strategy draws the same configs in every cell that shares a
  seed, so independent random draws = number of seeds (3). Read the per-seed ratios above as the conservative
  view; they all agree in direction.
- **Weaker baselines than a reviewer might want.** I compared against random search and a UCB bandit. There is no
  evolutionary / CMA-ES style local search and no off-the-shelf Bayesian optimizer, which are the natural
  competitors of a GP surrogate. Add one before claiming the surrogate is better than "adaptive search" in general.
- **No real large dataset.** This sandbox can't download data (OpenML is blocked) and can't install PyTorch, so
  there is no deep tabular model and the larger datasets are synthetic. Use `--csv` on real data, e.g. your own.
- **The "reference" is a 750-evaluation search, not the true minimum.** It is a yardstick; a strategy can beat it.
  A single cheapest-failure number moves by about 0.02 between searches on the same model; the failure rate over
  many configs is the steadier measure.
- **My first fix design made models worse.** Plain "retrain on the failures" increased failures on held-out tests,
  which is why fixes are now verified before acceptance, augmentation stays at or below the failing severity, and
  uses many separately-seeded draws. I did not tune these settings against the grid results.
- **I loosened the accept gate once, after seeing a result:** a 1.5-point clean-accuracy tolerance is below one
  sample on a 44-row validation set, so it is now `max(1.5 points, 1 sample)`.
- **The gate sees validation data only,** so accepted fixes can get worse on test (6 of 28 did).
- **Cost is not comparable across perturbation families** (severity scales are a design choice), and "important
  features" are relative to the current model, so before/after compares against a moving target.
- Tabular classifiers only; fixed perturbation menu.

## Layout

```
breakpoint_ml/
  perturbations.py   8 perturbation families
  space.py           Config, SearchSpace, encoding for the surrogate
  evaluator.py       cached, deterministic evaluation of one config on one model
  strategies.py      random / bandit / surrogate search
  benchmark.py       reference search + strategy comparison
  fix.py             propose -> verify -> accept loop
  builtin.py         datasets, model factories, CSV loader
  grid.py            multi-seed grid runner (parallel, resumable, records failures)
  aggregate.py       paired statistics, bootstrap CIs, grid charts and report
  report.py          single-run report.md + charts
  cli.py             demo / run / grid / aggregate
tests/               46 tests
runs/demo/           two-example report with charts
runs/grid/           108 cell JSONs + grid_report.md + charts
```

Use it on your own model: any fitted classifier with `predict` works with `Evaluator(model, X_val, y_val)`
(standardize features first), then `make_strategy("surrogate").run(ev, SearchSpace(), 150, np.random.default_rng(0))`
and `ev.failures()`.
