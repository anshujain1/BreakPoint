"""BreakPoint web app.  Run: python -m breakpoint_ml.server  ->  http://127.0.0.1:8000

Local use only. Uploaded .joblib/.pkl files are unpickled, which can run arbitrary code:
only load models you trained or trust.
"""
from __future__ import annotations

import dataclasses
import io
import os
import tempfile
import threading
import time
import uuid
import warnings

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_file, send_from_directory
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .builtin import MODELS, Splits, load_csv_splits, load_splits, make_model
from .evaluator import Evaluator
from .fix import find_and_fix, fixrun_to_dict
from .space import SearchSpace
from .strategies import make_strategy

app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), "web"), static_url_path="")
RUNS: dict = {}


class Standardized(ClassifierMixin, BaseEstimator):
    """Stress-test a model trained on raw features: perturb in standardized space, predict in raw space."""

    def __init__(self, model=None, mu=None, sd=None, names=None):
        self.model, self.mu, self.sd, self.names = model, mu, sd, names
        self.classes_ = getattr(model, "classes_", None)

    def fit(self, *a, **k):
        return self

    def predict(self, Z):
        X = np.asarray(Z) * self.sd + self.mu
        if hasattr(self.model, "feature_names_in_"):
            X = pd.DataFrame(X, columns=list(self.model.feature_names_in_))
        return np.asarray(self.model.predict(X)).ravel()


def _cfg(r):
    c = r.config
    return dict(i=r.index, family=c.family, target=c.target, severity=round(c.severity, 3),
                coverage=round(c.coverage, 3), cost=round(c.cost, 4), drop=round(r.drop, 4), failed=bool(r.failed))


def summarize(ev, names, meta):
    grid = {}
    for h in ev.history:
        g = grid.setdefault(h.config.family + "|" + h.config.target,
                            dict(family=h.config.family, target=h.config.target, n=0, fails=0, cheapest=None))
        g["n"] += 1
        if h.failed:
            g["fails"] += 1
            g["cheapest"] = h.config.cost if g["cheapest"] is None else min(g["cheapest"], h.config.cost)
    F = ev.failures()
    return dict(meta, clean=round(float(ev.clean_score), 4), tau=ev.tau, n_evals=ev.n_evals, n_failures=len(F),
                n_features=len(names), n_rows=int(len(ev.y)),
                breakpoint=_cfg(F[0]) if F else None,
                breakpoint_cols=[names[i] for i in ev.columns(F[0].config)][:12] if F else [],
                failures=[_cfg(x) for x in F[:15]], trace=[_cfg(h) for h in ev.history],
                grid=list(grid.values()), ranked_features=[names[i] for i in ev.rank[:12]])


def job(rid, p):
    r = RUNS[rid]
    try:
        warnings.filterwarnings("ignore")
        r["status"], r["p"] = "preparing", p
        seed = p["seed"]
        if p["dataset"]:
            sp, label, src = load_splits(p["dataset"], seed), p["model_name"], p["dataset"]
            est = make_model(p["model_name"], seed).fit(sp.X_tr, sp.y_tr)
            Z, y = sp.X_val, sp.y_val
            names = list(getattr(sp, dataclasses.fields(sp)[-1].name))
            r["splits"], r["factory"] = sp, (lambda: make_model(p["model_name"], seed))
        else:
            src = p["csv_name"]
            df = pd.read_csv(p["csv"]).dropna()
            if p["target"] not in df.columns:
                raise ValueError(f"Target column '{p['target']}' is not in the CSV.")
            if p["model_path"]:
                X, y = df.drop(columns=[p["target"]]), df[p["target"]].to_numpy()
                bad = [c for c in X if not pd.api.types.is_numeric_dtype(X[c])]
                if bad:
                    raise ValueError(f"Non-numeric columns {bad[:5]}: give the model numeric features it was trained on.")
                names, Xr = list(X.columns), X.to_numpy(float)
                mu, sd = Xr.mean(0), Xr.std(0)
                sd[sd == 0] = 1.0
                Z, model = (Xr - mu) / sd, joblib.load(p["model_path"])
                est, label = Standardized(model, mu, sd, names), type(model).__name__
                r["model"] = model
            else:
                sp = load_csv_splits(p["csv"], p["target"], seed)
                est, label = make_model(p["model_name"], seed).fit(sp.X_tr, sp.y_tr), p["model_name"]
                Z, y = sp.X_val, sp.y_val
                names = list(getattr(sp, dataclasses.fields(sp)[-1].name))
                r["splits"], r["factory"] = sp, (lambda: make_model(p["model_name"], seed))
        ev = Evaluator(est, Z, y, tau=p["tau"], seed=seed)
        r["ev"], r["status"] = ev, "searching"
        t = time.time()
        make_strategy("surrogate").run(ev, SearchSpace(), p["budget"], np.random.default_rng(seed))
        r["result"] = summarize(ev, names, dict(model=label, source=src, budget=p["budget"],
                                                seconds=round(time.time() - t, 1)))
        r["status"] = "done"
    except Exception as e:  # shown to the user verbatim
        r["status"], r["error"] = "error", f"{type(e).__name__}: {e}"


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/meta")
def meta():
    return jsonify(models=sorted(MODELS))


@app.post("/api/columns")
def columns():
    return jsonify(columns=list(pd.read_csv(request.files["csv"], nrows=0).columns))


@app.post("/api/runs")
def start():
    f, tmp, rid = request.form, tempfile.mkdtemp(), uuid.uuid4().hex[:8]
    p = dict(dataset=f.get("dataset") or None, model_name=f.get("model_name", "rf"), target=f.get("target"),
             budget=max(20, min(int(f.get("budget", 100)), 400)), tau=float(f.get("tau", 0.10)),
             seed=int(f.get("seed", 0)), csv=None, csv_name="", model_path=None)
    if "csv" in request.files and request.files["csv"].filename:
        p["csv"] = os.path.join(tmp, "data.csv")
        request.files["csv"].save(p["csv"])
        p["csv_name"] = request.files["csv"].filename
    if "model" in request.files and request.files["model"].filename:
        p["model_path"] = os.path.join(tmp, "model.joblib")
        request.files["model"].save(p["model_path"])
    if not p["dataset"] and not p["csv"]:
        return jsonify(error="Upload a CSV or pick a sample."), 400
    RUNS[rid] = dict(id=rid, status="queued", created=time.time(), budget=p["budget"],
                     name=f"{p['model_name'] if not p['model_path'] else 'uploaded model'} on {p['dataset'] or p['csv_name']}")
    threading.Thread(target=job, args=(rid, p), daemon=True).start()
    return jsonify(id=rid)


def _view(r, full=True):
    ev = r.get("ev")
    out = dict(id=r["id"], name=r["name"], status=r["status"], budget=r["budget"], created=r["created"],
               n_evals=ev.n_evals if ev else 0, n_failed=sum(h.failed for h in ev.history) if ev else 0,
               error=r.get("error"))
    f = r.get("fix")
    out["fix"] = f and dict(status=f["status"], log=f["log"][-8:], error=f.get("error"), result=f.get("result"),
                            downloadable="fixed" in r)
    out["need_train"] = r["status"] == "done" and "splits" not in r
    if full:
        out["result"] = r.get("result")
    return out


@app.get("/api/runs")
def runs():
    return jsonify([_view(r, False) for r in sorted(RUNS.values(), key=lambda r: -r["created"])])


@app.get("/api/runs/<rid>")
def run(rid):
    return (jsonify(_view(RUNS[rid])) if rid in RUNS else (jsonify(error="unknown run"), 404))


def upload_splits(train_csv, p, r):
    """Train rows come from the training CSV; the held-out CSV is split 50/50 into val (search + gate) and test."""
    tr, ho, t = pd.read_csv(train_csv).dropna(), pd.read_csv(p["csv"]).dropna(), p["target"]
    names = [c for c in ho.columns if c != t]
    miss = [c for c in names + [t] if c not in tr.columns]
    if miss:
        raise ValueError(f"The training CSV is missing columns: {miss[:5]}")
    Xt = tr[names].to_numpy(float)
    mu, sd = Xt.mean(0), Xt.std(0)
    sd[sd == 0] = 1.0
    Zh, yh = (ho[names].to_numpy(float) - mu) / sd, ho[t].to_numpy()
    try:
        Xv, Xe, yv, ye = train_test_split(Zh, yh, test_size=0.5, stratify=yh, random_state=p["seed"])
    except ValueError:
        Xv, Xe, yv, ye = train_test_split(Zh, yh, test_size=0.5, random_state=p["seed"])
    r["mu"], r["sd"] = mu, sd
    return Splits((Xt - mu) / sd, tr[t].to_numpy(), Xv, yv, Xe, ye, names)


def fix_job(rid, train_path):
    r = RUNS[rid]
    f = r["fix"]
    try:
        warnings.filterwarnings("ignore")
        p, seed = r["p"], r["p"]["seed"]
        if "splits" in r:
            sp, factory = r["splits"], r["factory"]
        else:
            sp, factory = upload_splits(train_path, p, r), (lambda: clone(r["model"]))
        run = find_and_fix(factory, sp, make_strategy("surrogate"), SearchSpace(), budget=min(p["budget"], 150),
                           rounds=1, tau=p["tau"], seed=seed, members=5, reference_evals=(400, 100),
                           log=lambda m: f["log"].append(m.strip()))
        d = fixrun_to_dict(run)
        a, b = d["rounds"][0], d["rounds"][-1]
        acc = b["recipe"] != "baseline"
        ref = [x["min_cost"] if x else None for x in (d["reference_before"], d["reference_after"])]
        f["result"] = dict(accepted=acc, recipe=b["recipe"].split("(")[0].replace("_", " "), decision=a["decision"],
                           candidates=a["candidates"], clean_test=[a["clean_test"], b["clean_test"]],
                           gentle_fail=[a["battery_low_cost_failure_rate"], b["battery_low_cost_failure_rate"]],
                           ref_cost=ref, n_train=[a["n_train"], b["n_train"]], n_val=len(sp.y_val), n_test=len(sp.y_te))
        if acc and r.get("model") is not None:
            sc = StandardScaler()
            sc.mean_, sc.scale_, sc.var_ = r["mu"], r["sd"], r["sd"] ** 2
            sc.n_features_in_, sc.n_samples_seen_ = len(r["mu"]), 1
            r["fixed"] = Pipeline([("scale", sc), ("model", run.models[-1])])
        f["status"] = "done"
    except Exception as e:
        f["status"], f["error"] = "error", f"{type(e).__name__}: {e}"


@app.post("/api/runs/<rid>/fix")
def fix(rid):
    r = RUNS.get(rid)
    if not r or r["status"] != "done":
        return jsonify(error="Run the stress test first."), 400
    if r.get("fix", {}).get("status") == "running":
        return jsonify(error="A fix search is already running."), 400
    path = None
    if "splits" not in r:
        fl = request.files.get("train")
        if not fl or not fl.filename:
            return jsonify(error="Upload the training CSV your model was trained on."), 400
        path = os.path.join(tempfile.mkdtemp(), "train.csv")
        fl.save(path)
    r.pop("fixed", None)
    r["fix"] = dict(status="running", log=[])
    threading.Thread(target=fix_job, args=(rid, path), daemon=True).start()
    return jsonify(id=rid)


@app.get("/api/runs/<rid>/fixed.joblib")
def fixed(rid):
    r = RUNS.get(rid)
    if not r or "fixed" not in r:
        return jsonify(error="No fixed model for this run."), 404
    buf = io.BytesIO()
    joblib.dump(r["fixed"], buf)
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name="breakpoint-fixed-model.joblib")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 8000)), threaded=True)
