# BreakPoint: hypothesis and architecture

## The problem

A classifier that scores well on a clean test set can still fall over under small, realistic input
changes: a noisy sensor, a missing field, a unit drift, a shifted population. Robustness testing tools
usually either (a) apply a fixed menu of perturbations at fixed strengths, or (b) run an expensive
adversarial attack. Neither answers the question a team actually has:

> **How gently can the inputs change before my model breaks, in which way, and can I make it stronger?**

BreakPoint treats that as a *search problem*. The model is a black box with a clean score. The search space is
"all the ways to perturb its inputs". The objective is the **cheapest perturbation that causes a failure**
(the model's *breakpoint*). Each evaluation costs one pass of the model over a validation set, so the number
of evaluations is the budget that matters.

## Definitions

A **stress test** is a point in a 4-part space:

| part | values |
|---|---|
| family | 8 perturbations: Gaussian noise, missing values, mean shift, scale drift, outlier spikes, coarse rounding, shared common-mode noise, feature-block shuffle |
| severity | 0.02 - 1.0, scaled per family (e.g. noise sigma = 3 x severity standard deviations) |
| coverage | 0.1 - 1.0, fraction of features touched |
| target | random features, the model's most important features, or its least important (by permutation importance) |

- **cost** = severity x coverage, in [0, 1]. A gentler perturbation has lower cost.
- A test **fails** when accuracy falls by at least `tau` (default 10 points) versus the clean score.
- **Breakpoint** = the lowest cost among failing tests found.
- **Utility** (the single objective every search strategy optimizes):
  `utility = min(drop / tau, 1) x (1 - cost)`. It is 1 for a failure that costs nothing, and shrinks both when
  the damage is under the threshold and when a large perturbation was needed to cause it.

## Hypotheses

**H1 (search).** The model's failure boundary in this space is structured enough (smooth in severity and
coverage, clustered by family and target) that a learned surrogate of "config -> accuracy drop" can find the
cheapest failures in far fewer model evaluations than uninformed or coarsely informed search.

**H2 (repair).** The failures an adaptive search finds are *informative for repair*: training on perturbed
copies of exactly those failures should reduce failures on unseen stress tests more than the same amount of
blind augmentation.

**H3 (verification).** Fixes cannot be trusted by construction. A propose -> verify -> accept loop, with the
decision made on validation data and the claim measured on untouched test data, avoids shipping fixes that
make the model worse.

### What the evidence says (108 cells: 6 models x 6 datasets x 3 seeds; see `runs/grid/grid_report.md`)

| hypothesis | verdict | numbers |
|---|---|---|
| H1 | **Supported.** | Surrogate's cheapest failure is 0.38x random's (95% CI 0.35-0.42) and 0.52x the bandit's; wins 106/108 and 100/108. Holds per seed (0.46 / 0.34 / 0.36), per model family and per dataset. Within 1.25x of a 750-evaluation reference search in 86/108 cells vs 2/108 for random. |
| H2 | **Not supported as stated.** | Targeted augmentation beat blind random augmentation on the validation gate in 54 cells, tied in 11, lost in 43. Fixes that were accepted reduced test failures by a mean 3.6 points, but part of that is generic noise-robust training, not learning from the failures. |
| H3 | **Partly supported.** | The first, unverified design made models worse on held-out tests. With the gate, 80 of 108 cells correctly left the model alone; of 28 accepted fixes, 21 helped on test, 6 hurt, 1 unchanged. Across all cells the gentle-config failure rate went from 20.2% to 19.3%. |

The honest summary: **BreakPoint is a strong, sample-efficient failure finder with a cautious, modest repair
step.** The finding side is the contribution; the repair side is an honest first version.

## Architecture

```
                        ┌──────────────────────────────────────────────┐
  fitted model  ──────► │ Evaluator  (evaluator.py)                     │
  validation data       │  cached, deterministic: Config -> Result      │
                        │  clean score, permutation-importance ranking, │
                        │  columns(cfg), perturb(X, cfg), evaluate(cfg) │
                        └───────▲──────────────────────────┬───────────┘
                                │ uses                      │ results (drop, failed, cost)
   ┌────────────────────────────┴──────┐        ┌───────────▼───────────────┐
   │ perturbations.py  (8 families)    │        │ strategies.py              │
   │ space.py  Config, SearchSpace,    │◄───────┤  random / bandit (UCB1) /  │
   │           sample, mutate, encode  │        │  surrogate (GP + UCB)      │
   └───────────────────────────────────┘        └───────────┬───────────────┘
                                                            │ failures found
                        ┌───────────────────────────────────▼───────────────┐
                        │ fix.py   propose -> verify -> accept              │
                        │  diagnose failures (target / family shares)       │
                        │  candidates: targeted_augmentation,               │
                        │              random_augmentation (control),       │
                        │              feature_subspace ensemble            │
                        │  gate on VAL: >=2 pts fewer gentle failures,      │
                        │               <=1.5 pts (or 1 sample) clean loss  │
                        │  report on TEST: never seen by search or gate     │
                        └───────────────────────────────────┬───────────────┘
                                                            │
   benchmark.py  reference search + strategy comparison     │
   grid.py       models x datasets x seeds, parallel,       ▼
                 resumable, one JSON per cell        report.py / aggregate.py
   builtin.py    datasets, model factories, CSV      markdown + charts, paired
                 loader                              statistics, bootstrap CIs
                                  cli.py:  demo | run | grid | aggregate
```

### Components and their contracts

- **`perturbations.py`**: pure functions `(X, columns, severity, rng) -> X'`. Severity 0 is the identity,
  inputs are never modified, only the selected columns change (all three properties are tested).
- **`space.py`**: `Config` is a frozen value (hashable, so results cache) whose RNG seed is derived from its
  fields, making every config reproducible. `SearchSpace` samples, mutates and encodes configs
  (one-hot family and target + severity + coverage = 13 dimensions) for the surrogate.
- **`evaluator.py`**: one fitted model plus one dataset. Same config always gives the same result; repeats are
  free and do not count against a budget; `fork()` gives a fresh history over the same model.
- **`strategies.py`**: three strategies behind one `run(evaluator, space, budget, rng)` interface.
  - *random*: uniform sampling, the baseline.
  - *bandit*: UCB1 over the 24 (family, target) arms with local refinement inside the best region of an arm.
    It learns which kinds of perturbation hurt but has no model of how severity and coverage interact.
  - *surrogate*: after a warm-up of random evaluations (max(12, 15% of budget)), fit a Gaussian process
    (ARD Matern-5/2 kernel plus noise) to accuracy drop, score about 1,500 candidate configs without running the
    model (1,200 random plus mutations of the best finds), run the one with the best optimistic utility
    `min((mu + sigma)/tau, 1) x (1 - cost)`, and repeat. An 8% pure-random rate guards against a wrong belief.
- **`fix.py`**: see the diagram. Data discipline is the core design rule: `train` fits models, `val` drives
  the search and the accept/reject gate, `test` is only used for the reported before/after numbers.
- **`benchmark.py` / `grid.py` / `aggregate.py`**: evaluation machinery. A big random-plus-refinement
  "reference" search gives a yardstick (not the true minimum). The grid runs one *cell* per
  (dataset, model, seed), in parallel, resumable, recording failures instead of hiding them; aggregation
  uses paired comparisons within cells and bootstrap confidence intervals.

## How the architecture is being completed

| stage | content | state |
|---|---|---|
| 1. Core | perturbation families, search space, deterministic cached evaluator | done, tested |
| 2. Search | random, bandit and GP-surrogate strategies; reference search | done, tested |
| 3. Repair | propose -> verify -> accept loop with candidate fixes and a validation gate | done; effect is small (see H2/H3) |
| 4. Evidence | multi-seed grid over 6 models (incl. XGBoost, LightGBM, CatBoost) x 6 datasets, paired statistics | done for 3 seeds; 5-seed run is resumable |
| 5. Strengthen the claim | an evolutionary / CMA-ES baseline and an off-the-shelf Bayesian optimizer to compete with the surrogate | **not done** (biggest gap in the evidence) |
| 6. Real data | run on at least one real dataset via `--csv` (OpenML and PyTorch were unreachable in the build sandbox) | **not done**, loader is in place |
| 7. Better repair | candidates driven by the diagnosis (e.g. train with feature dropout when failures concentrate on important features), plus a deep tabular model wrapper | **not done** |
| 8. Packaging | optional: a small dashboard on top of `results.json`, an `Evaluator` wrapper for non-sklearn models | **not done**, deliberately last |

## Known limitations

- Cost is not comparable across perturbation families; compare within a family or within a cell.
- "Important features" are relative to the current model, so before/after repair compares against a moving target.
- The reference search is a yardstick, and a single breakpoint number moves by about 0.02 between searches;
  failure rates over many configs are steadier.
- Confidence intervals are over cells that share models and datasets, and the random strategy shares its
  sample stream within a seed, so independent draws equal the number of seeds. Per-seed ratios are the
  conservative view.
- Tabular classifiers only; built-in and synthetic data only so far.
