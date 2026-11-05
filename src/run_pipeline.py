"""End-to-end reproducible pipeline.

    python -m src.run_pipeline            # generate data + run every experiment
    python -m src.run_pipeline --skip-data  # reuse data/ and only re-run experiments

Stages
------
1. Synthesise flows and extract features (``src.dataset``)
2. Main experiment: 4 models, stratified 70/30 split, 5-fold CV on train
3. Feature importance (permutation, per feature and per feature group)
4. Feature-group ablation
5. Early classification from the first N packets
6. Countermeasures (padding / timing jitter) vs. adaptive & non-adaptive attacker
7. Domain shift: broadband -> cellular network conditions
8. Figures + CSV/JSON results in ``results/``
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from . import config as C
from . import plotting as P
from .dataset import build_datasets, load_csv
from .defenses import DEFENSES
from .evaluate import confusion, fit_and_score, per_class_metrics, summary_metrics
from .features import FEATURE_GROUPS, FEATURE_NAMES
from .models import make_model, make_models

R = C.RESULTS_DIR
FULL_X = 600  # x-position used for "full flow" on the early-classification plot


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def split_ids(df: pd.DataFrame):
    """Stratified train/test split on flow_id; the same split is reused for
    every view (prefix / defended) so results are directly comparable."""
    tr, te = train_test_split(df["flow_id"], test_size=C.TEST_SIZE, stratify=df["label"], random_state=C.SEED)
    return set(tr), set(te)


def xy(df: pd.DataFrame, ids: set, cols=FEATURE_NAMES):
    sub = df[df["flow_id"].isin(ids)]
    return sub[cols], sub["label"].values, sub


# ---------------------------------------------------------------------------
def main_experiment(df, tr_ids, te_ids):
    """Train/evaluate all models; return metrics table and fitted models."""
    X_tr, y_tr, _ = xy(df, tr_ids)
    X_te, y_te, sub_te = xy(df, te_ids)
    cv = StratifiedKFold(C.CV_FOLDS, shuffle=True, random_state=C.SEED)
    rows, per_class, fitted, preds = [], [], {}, {}
    for name, model in make_models().items():
        log(f"  {name}: {C.CV_FOLDS}-fold CV ...")
        cv_scores = cross_val_score(clone(model), X_tr, y_tr, cv=cv, scoring="f1_macro", n_jobs=1)
        m, pred = fit_and_score(model, X_tr, y_tr, X_te, y_te)
        m.update(model=name, cv_f1_macro_mean=cv_scores.mean(), cv_f1_macro_std=cv_scores.std())
        rows.append(m)
        per_class.append(per_class_metrics(y_te, pred, name))
        fitted[name], preds[name] = model, pred
        cm = confusion(y_te, pred)
        cm.to_csv(R / f"confusion_matrix_{name}.csv")
        log(f"  {name}: test acc={m['accuracy']:.4f} macroF1={m['f1_macro']:.4f} "
            f"(CV {cv_scores.mean():.4f}±{cv_scores.std():.4f})")
    metrics = pd.DataFrame(rows)[["model", "accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted",
                                  "cv_f1_macro_mean", "cv_f1_macro_std", "train_time_s", "predict_us_per_flow"]]
    return metrics, pd.concat(per_class), fitted, preds, sub_te


def importance(model, name, X_te, y_te):
    """Permutation importance (macro-F1 drop) per feature and per group."""
    pi = permutation_importance(model, X_te, y_te, scoring="f1_macro", n_repeats=10, random_state=C.SEED, n_jobs=1)
    group_of = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}
    imp = pd.DataFrame({"feature": FEATURE_NAMES, "group": [group_of[f] for f in FEATURE_NAMES],
                        "importance_mean": pi.importances_mean, "importance_std": pi.importances_std})
    # group importance: permute all columns of a group with one shared row permutation
    rng = np.random.default_rng(C.SEED)
    base = f1_score(y_te, model.predict(X_te), average="macro")
    grows = []
    for g, cols in FEATURE_GROUPS.items():
        drops = []
        for _ in range(10):
            Xp = X_te.copy()
            Xp[cols] = X_te[cols].values[rng.permutation(len(X_te))]
            drops.append(base - f1_score(y_te, model.predict(Xp), average="macro"))
        grows.append({"group": g, "n_features": len(cols), "importance_mean": np.mean(drops),
                      "importance_std": np.std(drops)})
    return imp.sort_values("importance_mean", ascending=False), pd.DataFrame(grows)


def ablation(df, tr_ids, te_ids, model_name):
    rows = []
    for g, cols in FEATURE_GROUPS.items():
        rest = [f for f in FEATURE_NAMES if f not in cols]
        r = {"group": g, "n_features": len(cols)}
        for key, use in (("only", cols), ("without", rest)):
            X_tr, y_tr, _ = xy(df, tr_ids, use)
            X_te, y_te, _ = xy(df, te_ids, use)
            m, _ = fit_and_score(make_model(model_name), X_tr, y_tr, X_te, y_te)
            r[f"accuracy_{key}"], r[f"f1_{key}"] = m["accuracy"], m["f1_macro"]
        rows.append(r)
        log(f"  group {g:9s}: only={r['f1_only']:.4f} without={r['f1_without']:.4f}")
    return pd.DataFrame(rows)


def early_classification(df_full, tr_ids, te_ids, model_name):
    rows, pcs = [], []
    views = [(k, load_csv(C.DATA_DIR / "views" / f"prefix_{k}.csv")) for k in C.EARLY_PACKET_COUNTS]
    views.append(("full", df_full))
    for k, v in views:
        X_tr, y_tr, _ = xy(v, tr_ids)
        X_te, y_te, _ = xy(v, te_ids)
        m, pred = fit_and_score(make_model(model_name), X_tr, y_tr, X_te, y_te)
        num = FULL_X if k == "full" else k
        rows.append({"n_packets": k, "n_packets_num": num, **m})
        pc = per_class_metrics(y_te, pred, model_name)
        pc["n_packets_num"] = num
        pcs.append(pc)
        log(f"  first {k!s:>4} packets: acc={m['accuracy']:.4f} macroF1={m['f1_macro']:.4f}")
    return pd.DataFrame(rows), pd.concat(pcs)


def defense_costs_per_class(te_ids) -> pd.DataFrame:
    """Per-class byte overhead and median per-flow queueing delay of each defence."""
    rows = []
    for name in DEFENSES:
        v = load_csv(C.DATA_DIR / "views" / f"defense_{name}.csv")
        v = v[v["flow_id"].isin(te_ids)]
        for c in C.CLASSES:
            g = v[v["label"] == c]
            rows.append({"defense": name, "class": c,
                         "byte_overhead": float(g["n_bytes"].sum() / g["orig_bytes"].sum() - 1.0),
                         "median_delay_ms": 1e3 * float(g["mean_delay_s"].median()),
                         "p90_delay_ms": 1e3 * float(g["mean_delay_s"].quantile(0.9))})
    return pd.DataFrame(rows)


def countermeasures(df_full, tr_ids, te_ids, model_name, base_model):
    # without a defence the adaptive and non-adaptive attacker coincide
    base = summary_metrics(*_pred(base_model, df_full, te_ids))
    rows = [{"defense": "none", "byte_overhead": 0.0, "median_flow_mean_delay_ms": 0.0,
             **{f"{k}_adaptive": v for k, v in base.items()},
             **{f"{k}_nonadaptive": v for k, v in base.items()}}]
    for name in DEFENSES:
        v = load_csv(C.DATA_DIR / "views" / f"defense_{name}.csv")
        X_tr, y_tr, _ = xy(v, tr_ids)
        X_te, y_te, sub = xy(v, te_ids)
        m_ad, _ = fit_and_score(make_model(model_name), X_tr, y_tr, X_te, y_te)
        m_na = summary_metrics(y_te, base_model.predict(X_te))
        # aggregate byte overhead over all test flows (total padded / total original)
        # aggregate byte overhead over all test flows; note that bytes beyond
        # the 60 s observation window (queued traffic) are not counted
        overhead = float(sub["n_bytes"].sum() / sub["orig_bytes"].sum() - 1.0)
        delay_ms = 1e3 * float(sub["mean_delay_s"].median())
        rows.append({"defense": name, "byte_overhead": overhead, "median_flow_mean_delay_ms": delay_ms,
                     **{f"{k}_adaptive": m_ad[k] for k in ("accuracy", "precision_macro", "recall_macro",
                                                           "f1_macro", "f1_weighted")},
                     **{f"{k}_nonadaptive": val for k, val in m_na.items()}})
        log(f"  {name:20s}: adaptive F1={m_ad['f1_macro']:.4f} non-adaptive F1={m_na['f1_macro']:.4f} "
            f"overhead=+{100 * overhead:.1f}% delay={delay_ms:.1f}ms")
    return pd.DataFrame(rows)


def _pred(model, df, ids):
    X, y, _ = xy(df, ids)
    return y, model.predict(X)


def domain_shift(df_full, tr_ids, te_ids, model_name, base_model):
    cell_tr = load_csv(C.DATA_DIR / "flows_cellular_train.csv")
    cell_te = load_csv(C.DATA_DIR / "flows_cellular_test.csv")
    rows = []
    y, p = _pred(base_model, df_full, te_ids)
    rows.append({"train": "broadband", "test": "broadband", **summary_metrics(y, p)})
    p = base_model.predict(cell_te[FEATURE_NAMES])
    rows.append({"train": "broadband", "test": "cellular", **summary_metrics(cell_te["label"], p)})
    shift_pc = per_class_metrics(cell_te["label"], p, model_name)
    X_tr, y_tr, _ = xy(df_full, tr_ids)
    X_aug = pd.concat([X_tr, cell_tr[FEATURE_NAMES]])
    y_aug = np.concatenate([y_tr, cell_tr["label"].values])
    aug = make_model(model_name).fit(X_aug, y_aug)
    rows.append({"train": "broadband+cellular", "test": "cellular",
                 **summary_metrics(cell_te["label"], aug.predict(cell_te[FEATURE_NAMES]))})
    rows.append({"train": "broadband+cellular", "test": "broadband", **summary_metrics(*_pred(aug, df_full, te_ids))})
    out = pd.DataFrame(rows)
    for r in out.itertuples():
        log(f"  train={r.train:18s} test={r.test:9s} acc={r.accuracy:.4f} macroF1={r.f1_macro:.4f}")
    return out, shift_pc


# ---------------------------------------------------------------------------
def run(skip_data: bool = False) -> None:
    t_start = time.time()
    R.mkdir(exist_ok=True)
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    for stale in C.FIG_DIR.glob("*.png"):  # never mix figures from different runs
        stale.unlink()
    needed = [C.DATA_DIR / "flows.csv", C.DATA_DIR / "flows_cellular_test.csv",
              C.DATA_DIR / "views" / f"prefix_{C.EARLY_PACKET_COUNTS[0]}.csv",
              C.DATA_DIR / "views" / f"defense_{next(iter(DEFENSES))}.csv"]
    if not skip_data or not all(p.exists() for p in needed):
        log("Stage 1: generating synthetic flows ...")
        build_datasets()
    df = load_csv(C.DATA_DIR / "flows.csv")
    tr_ids, te_ids = split_ids(df)
    log(f"Dataset: {len(df)} flows, {len(FEATURE_NAMES)} features, train={len(tr_ids)} test={len(te_ids)}")
    stats = df.groupby("label").agg(flows=("flow_id", "size"), median_packets=("n_pkts", "median"),
                                    median_duration_s=("duration", "median"), median_bytes=("n_bytes", "median"))
    stats.loc[C.CLASSES].to_csv(R / "dataset_summary.csv")

    log("Stage 2: model comparison ...")
    metrics, per_class, fitted, preds, sub_te = main_experiment(df, tr_ids, te_ids)
    metrics.to_csv(R / "metrics.csv", index=False)
    per_class.to_csv(R / "per_class_metrics.csv", index=False)
    # model selection on CV score (never on the test set)
    best = metrics.sort_values("cv_f1_macro_mean", ascending=False).iloc[0]["model"]
    log(f"Best model by CV macro-F1: {best}")
    best_model = fitted[best]

    # error analysis: which sub-types get confused
    err = sub_te[["flow_id", "label", "subtype"]].copy()
    err["pred"] = preds[best]
    err["correct"] = err["label"] == err["pred"]
    by_sub = err.groupby(["label", "subtype"]).agg(n=("correct", "size"), accuracy=("correct", "mean")).reset_index()
    by_sub.to_csv(R / "errors_by_subtype.csv", index=False)
    err[~err["correct"]].to_csv(R / "misclassified_flows.csv", index=False)

    log("Stage 3: permutation importance ...")
    X_te, y_te, _ = xy(df, te_ids)
    imp, gimp = importance(best_model, best, X_te, y_te)
    imp.to_csv(R / "feature_importance_permutation.csv", index=False)
    gimp.to_csv(R / "feature_group_importance.csv", index=False)
    rf = fitted["RandomForest"]
    pd.DataFrame({"feature": FEATURE_NAMES, "importance": rf.feature_importances_}).sort_values(
        "importance", ascending=False).to_csv(R / "feature_importance_rf_impurity.csv", index=False)

    log("Stage 4: feature-group ablation ...")
    abl = ablation(df, tr_ids, te_ids, best)
    abl.to_csv(R / "ablation.csv", index=False)

    log("Stage 5: early classification ...")
    early, early_pc = early_classification(df, tr_ids, te_ids, best)
    early.to_csv(R / "early_classification.csv", index=False)
    early_pc.to_csv(R / "early_classification_per_class.csv", index=False)

    log("Stage 6: countermeasures ...")
    dfn = countermeasures(df, tr_ids, te_ids, best, best_model)
    dfn.to_csv(R / "defenses.csv", index=False)
    defense_costs_per_class(te_ids).to_csv(R / "defenses_cost_per_class.csv", index=False)

    log("Stage 7: domain shift ...")
    shift, shift_pc = domain_shift(df, tr_ids, te_ids, best, best_model)
    shift.to_csv(R / "domain_shift.csv", index=False)
    shift_pc.to_csv(R / "domain_shift_per_class.csv", index=False)

    log("Stage 8: figures ...")
    F = C.FIG_DIR
    P.plot_confusion(confusion(y_te, preds[best]), f"Confusion matrix — {best} (test set, n={len(y_te)})",
                     F / "confusion_matrix_best.png")
    worst = metrics.sort_values("f1_macro").iloc[0]["model"]
    P.plot_confusion(confusion(y_te, preds[worst]), f"Confusion matrix — {worst} (test set)",
                     F / f"confusion_matrix_{worst}.png")
    P.plot_feature_importance(imp, gimp, best, F / "feature_importance.png")
    P.plot_feature_distributions(df, ["byte_rate", "size_mean", "bwd_fwd_byte_ratio", "fwd_iat_regularity",
                                      "idle_frac", "frac_large", "burst_bytes_mean", "dir_change_rate"],
                                 F / "feature_distributions.png")
    P.plot_example_traces(C.DATA_DIR / "sample_traces.npz", F / "example_traces.png")
    P.plot_model_comparison(metrics, F / "model_comparison.png")
    P.plot_early(early, early_pc, best, F / "early_classification.png")
    P.plot_defenses(dfn, F / "defenses.png")
    P.plot_ablation(abl, F / "ablation.png")

    summary = {
        "n_flows": int(len(df)), "n_features": len(FEATURE_NAMES), "n_train": len(tr_ids), "n_test": len(te_ids),
        "best_model": best,
        "best_test_metrics": metrics.set_index("model").loc[best].to_dict(),
        "runtime_s": round(time.time() - t_start, 1),
    }
    (R / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"Done in {summary['runtime_s']} s. Results in {R}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-data", action="store_true", help="reuse existing data/ instead of regenerating")
    run(**vars(ap.parse_args()))
