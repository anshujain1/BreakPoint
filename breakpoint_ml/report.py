"""Markdown + PNG report for a BreakPoint run."""
from __future__ import annotations

import json
import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .benchmark import NO_FAILURE_COST, Reference, StrategyOutcome  # noqa: E402
from .fix import LOW_COST, FixRun, fixrun_to_dict  # noqa: E402
from .perturbations import FAMILIES, describe  # noqa: E402

# Reference palette (dataviz skill): categorical slots 1-3 validate under all-pairs.
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SERIES = {"random": "#2a78d6", "bandit": "#eb6834", "surrogate": "#1baf7a"}
FAIL_COLOR = "#eb6834"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=9)


def _fig(w=7.5, h=4.2):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=SURFACE)
    _style(ax)
    return fig, ax


def plot_benchmark(outcomes: Dict[str, StrategyOutcome], ref: Reference, path: str, title: str):
    fig, ax = _fig()
    ends = {}
    for name, o in outcomes.items():
        x = np.arange(1, o.curves.shape[1] + 1)
        m, s = o.curves.mean(0), o.curves.std(0)
        ax.fill_between(x, m - s, m + s, color=SERIES[name], alpha=0.15, linewidth=0)
        ax.plot(x, m, color=SERIES[name], linewidth=2, label=name)
        ends[name] = (x[-1], m[-1])
    # direct labels at the line ends, nudged apart so they never overlap
    ymax = min(0.5, max(ref.min_cost * 3, max(v[1] for v in ends.values()) * 1.3))
    gap, last = 0.045 * ymax / 0.5 * 1.0, None
    for name, (xe, ye) in sorted(ends.items(), key=lambda kv: kv[1][1]):
        ypos = ye if last is None else max(ye, last + gap)
        ax.annotate(name, (xe, ye), xytext=(xe + 4, ypos), textcoords="data", color=INK2,
                    fontsize=9, va="center",
                    arrowprops=dict(arrowstyle="-", color=GRID, linewidth=0.8, shrinkA=0, shrinkB=2))
        last = ypos
    ax.axhline(ref.min_cost, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.2)
    ax.annotate(f"reference ({ref.n_evals} evals): {ref.min_cost:.3f}", (1, ref.min_cost),
                xytext=(4, -12), textcoords="offset points", color=INK2, fontsize=8.5)
    ax.set_xlabel("evaluations spent", color=INK2)
    ax.set_ylabel("cheapest failure found (lower is better)", color=INK2)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.set_ylim(0, 0.5)
    ax.set_xlim(0, ends[next(iter(ends))][0] * 1.2)
    ax.legend(frameon=False, labelcolor=INK2, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_failure_map(history, path: str, title: str):
    fams = list(FAMILIES)
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.4), facecolor=SURFACE, sharex=True, sharey=True)
    for ax, fam in zip(axes.ravel(), fams):
        _style(ax)
        pts = [r for r in history if r.config.family == fam]
        ok = [r for r in pts if not r.failed]
        bad = [r for r in pts if r.failed]
        ax.scatter([r.config.severity for r in ok], [r.config.coverage for r in ok], s=22,
                   facecolors="none", edgecolors=MUTED, linewidths=0.9, label="held up")
        ax.scatter([r.config.severity for r in bad], [r.config.coverage for r in bad], s=26,
                   color=FAIL_COLOR, edgecolors=SURFACE, linewidths=0.8, label="failed")
        xs = np.linspace(0.02, 1, 100)
        ax.plot(xs, np.clip(LOW_COST / xs, 0, 1), color=GRID, linewidth=1.2)  # cost = LOW_COST
        ax.set_title(fam.replace("_", " "), loc="left", color=INK, fontsize=9.5)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
    for ax in axes[-1]:
        ax.set_xlabel("severity", color=INK2, fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("coverage", color=INK2, fontsize=9)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", frameon=False, labelcolor=INK2, fontsize=9, ncol=2)
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=11)
    fig.text(0.01, 0.01, f"grey line: cost = severity x coverage = {LOW_COST} (below/left = gentle)",
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_fix(run: FixRun, path: str, title: str):
    fig, ax = _fig(6.5, 3.8)
    rates = [100 * r.battery_low_cost_failure_rate for r in run.rounds]
    x = np.arange(len(rates))
    ax.bar(x, rates, width=0.5, color=SERIES["random"])
    for xi, r, st in zip(x, rates, run.rounds):
        ax.annotate(f"{r:.1f}%", (xi, r), xytext=(0, 4), textcoords="offset points", ha="center",
                    color=INK2, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"round {r.round}\n{r.recipe}" for r in run.rounds], fontsize=8)
    ax.set_ylabel("gentle configs that still break the model (%)", color=INK2)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.set_ylim(0, max(rates + [1]) * 1.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------
def _pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _cost(x):
    return "none found" if x is None or x >= NO_FAILURE_COST else f"{x:.3f}"


def _describe_failure(f: dict) -> str:
    tgt = {"important": "the model's most important features",
           "unimportant": "its least important features",
           "random": "randomly chosen features"}[f["target"]]
    return (f"{describe(f['family'], f['severity'])} on {100 * f['coverage']:.0f}% of features "
            f"({tgt})")


def section_markdown(sec: dict, out_dir: str, idx: int) -> str:
    name = f"{sec['dataset']} / {sec['model']}"
    tag = f"{sec['dataset']}_{sec['model']}"
    run: FixRun = sec["fix"]
    ref: Reference = sec["ref"]
    outcomes: Dict[str, StrategyOutcome] = sec["bench"]
    budget = sec["budget"]
    L: List[str] = [f"## {idx}. {name}", ""]

    # --- baseline -----------------------------------------------------------
    r0 = run.rounds[0]
    L += [f"Clean accuracy: **{r0.clean_test:.1%}** (test). A config *fails* when it costs at least "
          f"**{run.tau:.0%}** accuracy. In a large reference search on the validation split "
          f"({ref.n_evals} evals) **{ref.fail_fraction:.0%}** of all stress tests break the model; "
          f"the gentlest failure found costs **{ref.min_cost:.3f}** (cost = severity x coverage, "
          f"0 = untouched, 1 = everything at maximum).", ""]

    # --- search comparison -------------------------------------------------
    L += ["### Finding failures: strategy comparison", "",
          f"Each strategy gets {budget} model evaluations; {len(next(iter(outcomes.values())).curves)} "
          f"independent seeds. Lower final cost = a gentler failure discovered.", "",
          "| strategy | cheapest failure (mean +- sd) | evals to get within 0.05 of reference | "
          "failures found | seconds/run |", "|---|---|---|---|---|"]
    for n, o in outcomes.items():
        L.append(f"| {n} | {o.final_costs.mean():.3f} +- {o.final_costs.std():.3f} | "
                 f"{np.mean(o.evals_to_near_ref):.0f}{'+' if max(o.evals_to_near_ref) > budget else ''} | "
                 f"{np.mean(o.n_failures):.1f} | {np.mean(o.seconds):.1f} |")
    L += ["", "(A `+` means at least one seed never got within 0.05 of the reference inside the budget; "
          "those runs count as budget + 1.)", "",
          f"![strategy comparison]({tag}_benchmark.png)", ""]

    # --- failure map -------------------------------------------------------
    L += ["### Where it breaks", "", f"![failure map]({tag}_failure_map.png)", "",
          "Cheapest failures found on the validation split (baseline model):", "",
          "| cost | accuracy drop | stress test |", "|---|---|---|"]
    for f in r0.top_failures[:6]:
        L.append(f"| {f['cost']:.3f} | {100 * f['drop']:.1f} pts | {_describe_failure(f)} |")
    L.append("")
    diag = [(k, v) for k, v in r0.diagnosis.items() if k.startswith("target:")]
    if diag:
        L.append("Of the 20 cheapest failures: " + ", ".join(
            f"{100 * v:.0f}% hit {k.split(':')[1]} features" for k, v in diag) + ".")
        fam = [(k.split(":")[1], v) for k, v in r0.diagnosis.items() if k.startswith("family:")][:3]
        L.append("Most common perturbation families: " + ", ".join(
            f"{k.replace('_', ' ')} ({100 * v:.0f}%)" for k, v in fam) + ".")
        L.append("")

    # --- find-and-fix --------------------------------------------------------
    L += ["### Fixing: propose, verify, accept", "",
          "Candidate fixes are trained each round and scored on a *validation* gate "
          f"(failure rate over gentle configs, cost <= {LOW_COST}). One is kept only if it clears the "
          "gate; otherwise the model is left alone.", ""]
    for st in run.rounds:
        if not st.candidates and not st.decision:
            continue
        L += [f"**Round {st.round}** (model: {st.recipe})", "",
              "| candidate | gate failure rate (val) | clean acc (val) | passed gate | kept |",
              "|---|---|---|---|---|"]
        for c in st.candidates:
            L.append(f"| {c.name} | {_pct(c.gate_failure_rate)} | {_pct(c.clean_val)} | "
                     f"{'yes' if c.eligible else 'no'} | {'**yes**' if c.accepted else 'no'} |")
        L += ["", st.decision, ""]
    first, last = run.rounds[0], run.rounds[-1]
    L += ["### Result on the untouched test split", "",
          "| | before | after |", "|---|---|---|",
          f"| clean accuracy | {_pct(first.clean_test)} | {_pct(last.clean_test)} |",
          f"| gentle configs (cost <= {LOW_COST}) that break the model | "
          f"{_pct(first.battery_low_cost_failure_rate)} | {_pct(last.battery_low_cost_failure_rate)} |",
          f"| all battery configs that break the model | {_pct(first.battery_failure_rate)} | "
          f"{_pct(last.battery_failure_rate)} |",
          f"| mean accuracy under the battery | {_pct(first.battery_mean_score)} | "
          f"{_pct(last.battery_mean_score)} |"]
    if run.reference_before and run.reference_after:
        L.append(f"| cheapest failure found (independent {run.reference_before.n_evals}-eval search) | "
                 f"{run.reference_before.min_cost:.3f} | {run.reference_after.min_cost:.3f} |")
    L += ["", f"Final model: **{last.recipe}**.", "", f"![fix result]({tag}_fix.png)", ""]
    return "\n".join(L)


def write_report(sections: List[dict], out_dir: str, meta: dict) -> str:
    os.makedirs(out_dir, exist_ok=True)
    parts = ["# BreakPoint report", "",
             f"Stress-testing {len(sections)} model(s). Failure = accuracy drop >= "
             f"{meta['tau']:.0%}; search budget {meta['budget']} evaluations; "
             f"{meta['seeds']} seeds per strategy; fixes verified on validation and reported on test.", ""]
    dump = {}
    for i, sec in enumerate(sections, 1):
        tag = f"{sec['dataset']}_{sec['model']}"
        plot_benchmark(sec["bench"], sec["ref"], os.path.join(out_dir, f"{tag}_benchmark.png"),
                       f"{sec['dataset']} / {sec['model']}: cheapest failure vs evaluations")
        plot_failure_map(sec["fix"].history_per_round[0], os.path.join(out_dir, f"{tag}_failure_map.png"),
                         f"{sec['dataset']} / {sec['model']}: every stress test the search ran")
        plot_fix(sec["fix"], os.path.join(out_dir, f"{tag}_fix.png"),
                 f"{sec['dataset']} / {sec['model']}: gentle-config failures on test")
        parts.append(section_markdown(sec, out_dir, i))
        dump[tag] = dict(
            fix=fixrun_to_dict(sec["fix"]),
            reference=dict(min_cost=sec["ref"].min_cost, fail_fraction=sec["ref"].fail_fraction,
                           n_evals=sec["ref"].n_evals),
            benchmark={n: dict(final_cost_mean=float(o.final_costs.mean()),
                               final_cost_sd=float(o.final_costs.std()),
                               evals_to_near_ref=[int(v) for v in o.evals_to_near_ref],
                               failures_found=[int(v) for v in o.n_failures],
                               seconds=[float(v) for v in o.seconds])
                       for n, o in sec["bench"].items()})
    parts += ["## Limitations", "",
              "- **Cost is a design choice.** Severity is normalized per perturbation family, so a cost "
              "of 0.1 in one family is not physically equivalent to 0.1 in another. Compare costs "
              "within a family, or across families only as a rough ordering.",
              "- **'Important' features are model-relative.** Targeted configs hit whichever features "
              "the *current* model relies on most, so after a fix the same config attacks different "
              "features. That is intended (an adaptive attacker would do the same) but it means "
              "before/after numbers compare against a moving target.",
              "- **Small splits are noisy.** With a few hundred rows, one sample is a fraction of a "
              "point; differences of 1-3 points are within noise. The accept gate is tolerant of one "
              "validation sample for that reason.",
              "- **The reference is a bound, not the truth.** Its cheapest failure is the best of a "
              "finite search; a strategy can beat it. The breakpoint is a minimum over what was "
              "searched, so it can shift between search seeds on the *same* model.",
              "- **Search seeds matter.** Re-running the search on an unchanged model gave cheapest "
              "failures that differed by roughly 0.01-0.04 in our runs. Treat a single breakpoint "
              "number as an estimate and use the gentle-config failure rate for before/after claims.",
              "- **Tabular classifiers only.** Perturbations act on standardized numeric features.", ""]
    path = os.path.join(out_dir, "report.md")
    with open(path, "w") as f:
        f.write("\n".join(parts))
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(dump, f, indent=2, default=float)
    return path
