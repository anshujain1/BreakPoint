"""Aggregate a grid of cells into paired statistics, charts and a markdown summary."""
from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .benchmark import NO_FAILURE_COST  # noqa: E402
from .grid import SEARCHERS, load_cells  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
COLORS = {"random": "#2a78d6", "bandit": "#eb6834", "surrogate": "#1baf7a"}
HEADROOM = 0.05    # a cell has "headroom" if >= 5% of gentle test configs break the baseline model


def boot_ci(x: Sequence[float], stat=np.mean, n: int = 4000, seed: int = 0, level: float = 0.95):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    stats = np.array([stat(x[i]) for i in idx])
    lo, hi = np.percentile(stats, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(stat(x)), float(lo), float(hi)


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=9)


def split_cells(cells: List[dict]):
    ok = [c for c in cells if c.get("ok")]
    bad = [c for c in cells if not c.get("ok")]
    return ok, bad


# ---------------------------------------------------------------------------
# search comparison
# ---------------------------------------------------------------------------
def search_stats(ok: List[dict]) -> dict:
    """Paired comparisons. Only cells where failures exist (reference found one) are meaningful."""
    valid = [c for c in ok if c["ref"]["fail_fraction"] > 0]
    out = dict(n_cells=len(ok), n_valid=len(valid), n_no_failures=len(ok) - len(valid), per_strategy={},
               vs={})
    for s in SEARCHERS:
        costs = np.array([c["search"][s]["final_cost"] for c in valid])
        out["per_strategy"][s] = dict(
            found_none=int(np.sum(costs >= NO_FAILURE_COST)),
            mean_failures=float(np.mean([c["search"][s]["n_failures"] for c in valid])) if valid else float("nan"),
            median_cost=float(np.median(costs)) if len(costs) else float("nan"),
            mean_seconds=float(np.mean([c["search"][s]["seconds"] for c in valid])) if valid else float("nan"))
    for other in ("random", "bandit"):
        a = np.array([c["search"]["surrogate"]["final_cost"] for c in valid])
        b = np.array([c["search"][other]["final_cost"] for c in valid])
        wins, ties, losses = int(np.sum(a < b - 1e-9)), int(np.sum(np.abs(a - b) <= 1e-9)), int(np.sum(a > b + 1e-9))
        logr = np.log(a / b)
        gm, lo, hi = boot_ci(logr)
        per_seed = {}
        for sd in sorted({c["seed"] for c in valid}):
            lr = [np.log(c["search"]["surrogate"]["final_cost"] / c["search"][other]["final_cost"])
                  for c in valid if c["seed"] == sd]
            per_seed[sd] = float(np.exp(np.mean(lr)))
        out["vs"][other] = dict(wins=wins, ties=ties, losses=losses, gm_ratio=float(np.exp(gm)),
                                lo=float(np.exp(lo)), hi=float(np.exp(hi)), per_seed=per_seed)
    # random vs bandit, to keep the story honest
    a = np.array([c["search"]["bandit"]["final_cost"] for c in valid])
    b = np.array([c["search"]["random"]["final_cost"] for c in valid])
    gm, lo, hi = boot_ci(np.log(a / b))
    out["vs"]["bandit_vs_random"] = dict(wins=int(np.sum(a < b - 1e-9)), ties=int(np.sum(np.abs(a - b) <= 1e-9)),
                                         losses=int(np.sum(a > b + 1e-9)), gm_ratio=float(np.exp(gm)),
                                         lo=float(np.exp(lo)), hi=float(np.exp(hi)))
    # per-model breakdown of surrogate/random ratio
    by_model = defaultdict(list)
    for c in valid:
        by_model[c["model"]].append(np.log(c["search"]["surrogate"]["final_cost"] / c["search"]["random"]["final_cost"]))
    out["by_model"] = {m: dict(n=len(v), gm_ratio=float(np.exp(np.mean(v))),
                               wins=int(np.sum(np.array(v) < -1e-9))) for m, v in sorted(by_model.items())}
    by_ds = defaultdict(list)
    for c in valid:
        by_ds[c["dataset"]].append(np.log(c["search"]["surrogate"]["final_cost"] / c["search"]["random"]["final_cost"]))
    out["by_dataset"] = {d: dict(n=len(v), gm_ratio=float(np.exp(np.mean(v))),
                                 wins=int(np.sum(np.array(v) < -1e-9))) for d, v in sorted(by_ds.items())}
    out["valid"] = valid
    return out


# ---------------------------------------------------------------------------
# fix results
# ---------------------------------------------------------------------------
def fix_stats(ok: List[dict]) -> dict:
    rows = []
    for c in ok:
        r0, rl = c["fix"]["rounds"][0], c["fix"]["rounds"][-1]
        accepted = rl["recipe"] != "baseline"
        cands = {x["name"]: x for x in r0.get("candidates", [])}
        rows.append(dict(
            dataset=c["dataset"], model=c["model"], seed=c["seed"], accepted=accepted, recipe=rl["recipe"],
            base_low=r0["battery_low_cost_failure_rate"], final_low=rl["battery_low_cost_failure_rate"],
            base_clean=r0["clean_test"], final_clean=rl["clean_test"],
            base_all=r0["battery_failure_rate"], final_all=rl["battery_failure_rate"],
            targeted_gate=cands.get("targeted_augmentation", {}).get("gate_failure_rate"),
            random_gate=cands.get("random_augmentation", {}).get("gate_failure_rate"),
            subspace_gate=cands.get("feature_subspace", {}).get("gate_failure_rate")))
    for r in rows:
        r["d_low"] = r["final_low"] - r["base_low"]
        r["d_clean"] = r["final_clean"] - r["base_clean"]
        r["headroom"] = r["base_low"] >= HEADROOM
    acc = [r for r in rows if r["accepted"]]
    head = [r for r in rows if r["headroom"]]
    both = [r for r in rows if r["targeted_gate"] is not None and r["random_gate"] is not None]
    out = dict(rows=rows, n=len(rows), n_accepted=len(acc), n_headroom=len(head),
               n_head_accepted=len([r for r in head if r["accepted"]]))
    out["all_cells_d_low"] = boot_ci([r["d_low"] for r in rows])
    out["headroom_d_low"] = boot_ci([r["d_low"] for r in head])
    out["accepted_d_low"] = boot_ci([r["d_low"] for r in acc])
    out["accepted_d_clean"] = boot_ci([r["d_clean"] for r in acc])
    out["accepted_better"] = int(sum(r["d_low"] < -1e-9 for r in acc))
    out["accepted_worse"] = int(sum(r["d_low"] > 1e-9 for r in acc))
    out["accepted_same"] = len(acc) - out["accepted_better"] - out["accepted_worse"]
    out["targeted_vs_random"] = dict(
        n=len(both), wins=sum(r["targeted_gate"] < r["random_gate"] - 1e-9 for r in both),
        ties=sum(abs(r["targeted_gate"] - r["random_gate"]) <= 1e-9 for r in both),
        losses=sum(r["targeted_gate"] > r["random_gate"] + 1e-9 for r in both))
    kinds = defaultdict(int)
    for r in acc:
        kinds[r["recipe"].split("(")[0].split(" + ")[0]] += 1
    out["accepted_kinds"] = dict(kinds)
    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)
    out["by_model"] = {m: dict(n=len(v), headroom=sum(x["headroom"] for x in v), accepted=sum(x["accepted"] for x in v),
                               mean_d_low=float(np.mean([x["d_low"] for x in v])),
                               mean_d_low_head=float(np.mean([x["d_low"] for x in v if x["headroom"]]))
                               if any(x["headroom"] for x in v) else float("nan"))
                       for m, v in sorted(by_model.items())}
    return out


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------
def plot_convergence(valid: List[dict], path: str):
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)
    _style(ax)
    budget = len(valid[0]["search"]["random"]["curve"])
    x = np.arange(1, budget + 1)
    ends = {}
    for s in SEARCHERS:
        mat = np.array([np.array(c["search"][s]["curve"]) / max(c["ref"]["min_cost"], 1e-3) for c in valid])
        med = np.median(mat, axis=0)
        lo, hi = np.percentile(mat, [25, 75], axis=0)
        ax.fill_between(x, lo, hi, color=COLORS[s], alpha=0.12, linewidth=0)
        ax.plot(x, med, color=COLORS[s], linewidth=2, label=s)
        ends[s] = med[-1]
    ax.set_yscale("log")
    ax.axhline(1.0, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.2)
    ax.annotate("reference search (1x)", (2, 1.0), xytext=(2, 6), textcoords="offset points", color=INK2, fontsize=8.5)
    last = None
    for s, v in sorted(ends.items(), key=lambda kv: kv[1]):
        ypos = v if last is None else max(v, last * 1.12)
        ax.annotate(s, (budget, v), xytext=(budget + 3, ypos), color=INK2, fontsize=9, va="center",
                    arrowprops=dict(arrowstyle="-", color=GRID, linewidth=0.8, shrinkA=0, shrinkB=2))
        last = ypos
    ax.set_xlim(0, budget * 1.18)
    ax.set_ylim(0.8, 30)
    ax.set_xlabel("model evaluations spent", color=INK2)
    ax.set_ylabel("cheapest failure / reference (median, IQR band)", color=INK2)
    ax.set_title(f"All {len(valid)} cells: how fast each strategy approaches the reference", loc="left",
                 color=INK, fontsize=11)
    ax.legend(frameon=False, labelcolor=INK2, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_paired(valid: List[dict], path: str):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), facecolor=SURFACE)
    models = sorted({c["model"] for c in valid})
    markers = dict(zip(models, "osD^vP*X"))
    for ax, other in zip(axes, ("random", "bandit")):
        _style(ax)
        lim = (0.015, 1.05)
        ax.plot(lim, lim, color=MUTED, linewidth=1)
        for m in models:
            pts = [(c["search"][other]["final_cost"], c["search"]["surrogate"]["final_cost"])
                   for c in valid if c["model"] == m]
            ax.scatter(*zip(*pts), s=26, color=COLORS["surrogate"], alpha=0.7, marker=markers[m],
                       edgecolors=SURFACE, linewidths=0.5, label=m)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel(f"{other}: cheapest failure", color=INK2)
        ax.set_ylabel("surrogate: cheapest failure", color=INK2)
        ax.set_title(f"surrogate vs {other}, one dot per cell", loc="left", color=INK, fontsize=10.5)
        ax.annotate("surrogate better", (0.55, 0.05), xycoords="axes fraction", color=INK2, fontsize=9)
    axes[0].legend(frameon=False, labelcolor=INK2, fontsize=8, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_fix(rows: List[dict], path: str):
    fig, ax = plt.subplots(figsize=(8, 4.4), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="x", visible=False)
    models = sorted({r["model"] for r in rows})
    rng = np.random.default_rng(0)
    for i, m in enumerate(models):
        for r in (x for x in rows if x["model"] == m):
            jitter = rng.uniform(-0.18, 0.18)
            if r["accepted"]:
                ax.scatter(i + jitter, 100 * r["d_low"], s=34, color=COLORS["surrogate"], edgecolors=SURFACE,
                           linewidths=0.6, zorder=3)
            else:
                ax.scatter(i + jitter, 0.0, s=22, facecolors="none", edgecolors=MUTED, linewidths=0.9, zorder=3)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, color=INK2)
    ax.set_ylabel("change in gentle-config failure rate on test (points)\nnegative = fix helped", color=INK2)
    ax.set_title("Each dot is one cell: filled = a fix was accepted, hollow = model left unchanged",
                 loc="left", color=INK, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------
def _ci(t, pct=False, digits=3):
    m, lo, hi = t
    if pct:
        return f"{100 * m:+.1f} pts (95% CI {100 * lo:+.1f} to {100 * hi:+.1f})"
    return f"{m:.{digits}f} (95% CI {lo:.{digits}f} to {hi:.{digits}f})"


def write_summary(out_dir: str, models: Sequence[str] | None = None) -> str:
    cells = load_cells(out_dir)
    # keep only seeds for which every (dataset, model) combination finished, so all seeds weigh equally
    combos = {(c["dataset"], c["model"]) for c in cells}
    per_seed = defaultdict(int)
    for c in cells:
        per_seed[c["seed"]] += 1
    full = {sd for sd, n in per_seed.items() if n == len(combos)}
    dropped = sorted(set(per_seed) - full)
    cells = [c for c in cells if c["seed"] in full]
    ok, bad = split_cells(cells)
    if not ok:
        raise SystemExit("no finished cells found")
    S, F = search_stats(ok), fix_stats(ok)
    valid = S.pop("valid")
    plot_convergence(valid, os.path.join(out_dir, "grid_convergence.png"))
    plot_paired(valid, os.path.join(out_dir, "grid_paired.png"))
    plot_fix(F["rows"], os.path.join(out_dir, "grid_fix.png"))

    ds = sorted({c["dataset"] for c in ok})
    ms = sorted({c["model"] for c in ok})
    sd = sorted({c["seed"] for c in ok})
    L = ["# BreakPoint: multi-seed benchmark", "",
         f"**{len(ok)} cells** = {len(ms)} models x {len(ds)} datasets x {len(sd)} seeds "
         f"(models: {', '.join(ms)}; datasets: {', '.join(ds)}). "
         f"{len(bad)} cell(s) failed to run."
         + (f" (Seeds {dropped} were incomplete and are excluded.)" if dropped else "") + (" Failed cells: " + ", ".join(
             f"{b['dataset']}/{b['model']}/seed{b['seed']} ({b['error'][:60]})" for b in bad) if bad else ""),
         "", f"Search budget {ok[0]['budget']} evaluations per strategy per cell; failure = accuracy drop "
         f">= {ok[0]['tau']:.0%}. The seed changes the split, the model's init and the search RNG "
         "(synthetic datasets themselves are fixed). Every number below is a paired comparison within a cell.", "",
         "## 1. Finding failures: does the surrogate search find gentler ones?", "",
         f"{S['n_valid']} of {S['n_cells']} cells contain failures to find"
         + (f" ({S['n_no_failures']} cells excluded: the reference search found no failure at all)."
            if S["n_no_failures"] else "."), "",
         "| surrogate vs | cheapest failure ratio (geo-mean) | surrogate wins / ties / losses |", "|---|---|---|"]
    for o, label in (("random", "random"), ("bandit", "bandit")):
        v = S["vs"][o]
        L.append(f"| {label} | **{v['gm_ratio']:.2f}** (95% CI {v['lo']:.2f} to {v['hi']:.2f}) | "
                 f"{v['wins']} / {v['ties']} / {v['losses']} |")
    v = S["vs"]["bandit_vs_random"]
    ps = lambda o: ", ".join(f"seed {k}: {v:.2f}" for k, v in S["vs"][o]["per_seed"].items())
    L += ["", f"Per-seed geometric-mean ratio, surrogate vs random: {ps('random')}. Vs bandit: {ps('bandit')}. "
          "(The random strategy draws the same configs in every cell that shares a seed, so the number of "
          "*independent* random draws is the number of seeds; read the per-seed ratios as the conservative "
          "view and the confidence intervals above as optimistic.)", "",
          "A ratio below 1 means the surrogate's cheapest failure costs less (it found a gentler failure). "
          f"For reference, bandit vs random: ratio {v['gm_ratio']:.2f} (95% CI {v['lo']:.2f} to {v['hi']:.2f}), "
          f"bandit wins/ties/losses {v['wins']} / {v['ties']} / {v['losses']}.", "",
          "| strategy | median cheapest failure | cells with no failure found | failing configs found (mean) | "
          "seconds per search |", "|---|---|---|---|---|"]
    for s in SEARCHERS:
        p = S["per_strategy"][s]
        L.append(f"| {s} | {p['median_cost']:.3f} | {p['found_none']} | {p['mean_failures']:.1f} | {p['mean_seconds']:.1f} |")
    L += ["", "![convergence](grid_convergence.png)", "", "![paired](grid_paired.png)", "",
          "Surrogate / random ratio by model and by dataset (below 1 = surrogate found a gentler failure):", "",
          "| model | cells | ratio | wins |", "|---|---|---|---|"]
    for m, v in S["by_model"].items():
        L.append(f"| {m} | {v['n']} | {v['gm_ratio']:.2f} | {v['wins']}/{v['n']} |")
    L += ["", "| dataset | cells | ratio | wins |", "|---|---|---|---|"]
    for d, v in S["by_dataset"].items():
        L.append(f"| {d} | {v['n']} | {v['gm_ratio']:.2f} | {v['wins']}/{v['n']} |")

    L += ["", "## 2. Fixing: does propose-verify-accept help on held-out test data?", "",
          f"Cells with headroom (at least {HEADROOM:.0%} of gentle test configs break the baseline): "
          f"**{F['n_headroom']} of {F['n']}**. A fix was accepted in **{F['n_accepted']}** cells "
          f"({F['n_head_accepted']} of them with headroom). Accepted fixes: "
          + (", ".join(f"{k} x{v}" for k, v in F["accepted_kinds"].items()) or "none") + ".", "",
          "| change in gentle-config failure rate on test | mean |", "|---|---|",
          f"| all cells (unchanged models count as 0) | {_ci(F['all_cells_d_low'], pct=True)} |",
          f"| cells with headroom | {_ci(F['headroom_d_low'], pct=True)} |",
          f"| cells where a fix was accepted | {_ci(F['accepted_d_low'], pct=True)} |",
          f"| clean accuracy, accepted cells | {_ci(F['accepted_d_clean'], pct=True)} |", "",
          f"Of the {F['n_accepted']} accepted fixes, **{F['accepted_better']} reduced** gentle-config failures on "
          f"test, **{F['accepted_worse']} increased** them, {F['accepted_same']} were unchanged. "
          "The gate sees only validation data, so accepted fixes that get worse on test are the price of "
          "small validation sets.", ""]
    t = F["targeted_vs_random"]
    L += [f"Targeted vs blind random augmentation, same data volume, on the validation gate "
          f"({t['n']} cells where both were tried): targeted lower in {t['wins']}, tied in {t['ties']}, "
          f"higher in {t['losses']}.", "", "![fix](grid_fix.png)", "",
          "| model | cells | with headroom | fix accepted | mean change, all (pts) | mean change, headroom (pts) |",
          "|---|---|---|---|---|---|"]
    for m, v in F["by_model"].items():
        hd = "n/a" if np.isnan(v["mean_d_low_head"]) else f"{100 * v['mean_d_low_head']:+.1f}"
        L.append(f"| {m} | {v['n']} | {v['headroom']} | {v['accepted']} | {100 * v['mean_d_low']:+.1f} | {hd} |")
    L += ["", "## Caveats", "",
          "- Cells are not independent samples of 'all ML problems': they share datasets and models, "
          "and the synthetic datasets are fixed across seeds. Confidence intervals bootstrap over cells, "
          "so they describe this grid, not a population of real-world tasks.",
          "- Cost (severity x coverage) is not comparable across perturbation families; ratios are within a cell.",
          "- All data here is built-in or synthetic. Run `python -m breakpoint_ml run --csv ... --target ...` on "
          "real data before claiming anything beyond this grid.", ""]
    path = os.path.join(out_dir, "grid_report.md")
    with open(path, "w") as f:
        f.write("\n".join(L))
    return path
