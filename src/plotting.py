"""Figures for the results section.

Colour use follows one rule per job: a fixed categorical order for the six
traffic classes (identity), a single blue ramp for magnitudes (confusion
matrix), and neutral ink for text.  Every categorical chart also labels
classes on an axis or legend, so colour is never the only cue.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .config import CLASSES  # noqa: E402

# ---- palette -------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
CLASS_COLORS = dict(zip(CLASSES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # first three slots: safe for any pairing
BLUE_RAMP = LinearSegmentedColormap.from_list(
    "blue_seq", ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
PRETTY = {"file_transfer": "file transfer", "voip": "VoIP"}


def pretty(c: str) -> str:
    return PRETTY.get(c, c)


plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titleweight": "bold",
    "axes.titlesize": 11, "axes.labelsize": 9.5, "xtick.color": INK_2, "ytick.color": INK_2,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 8.5, "font.size": 9.5, "figure.dpi": 100,
})


def _save(fig, path: Path):
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
def plot_confusion(cm: pd.DataFrame, title: str, path: Path) -> None:
    """Row-normalised confusion matrix; each cell shows recall % and count."""
    counts = cm.values
    norm = counts / counts.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.grid(False)
    im = ax.imshow(norm, cmap=BLUE_RAMP, vmin=0, vmax=1)
    labels = [pretty(c) for c in cm.index]
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            if counts[i, j] == 0:
                continue
            color = "white" if norm[i, j] > 0.55 else INK
            ax.text(j, i, f"{100 * norm[i, j]:.1f}%\n({counts[i, j]})", ha="center", va="center",
                    fontsize=7.5, color=color)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Share of true class (recall)")
    cb.outline.set_visible(False)
    ax.set_title(title, loc="left")
    _save(fig, path)


def plot_feature_importance(imp: pd.DataFrame, group_imp: pd.DataFrame, model: str, path: Path, top: int = 20):
    """Permutation importance: top individual features and whole groups."""
    imp = imp.sort_values("importance_mean", ascending=False).head(top).iloc[::-1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5.6), gridspec_kw={"width_ratios": [2.2, 1]})
    a1.barh(range(len(imp)), imp["importance_mean"], xerr=imp["importance_std"], color=SERIES[0], height=0.7,
            error_kw={"ecolor": INK_2, "elinewidth": 0.8, "capsize": 0})
    a1.set_yticks(range(len(imp)), [f"{f}  ({g})" for f, g in zip(imp["feature"], imp["group"])])
    a1.set_xlabel("Drop in macro-F1 when the feature is permuted")
    a1.set_title(f"Top-{top} features ({model}, test set)", loc="left")
    a1.grid(axis="y", visible=False)
    g = group_imp.sort_values("importance_mean")
    a2.barh(g["group"], g["importance_mean"], xerr=g["importance_std"], color=SERIES[0], height=0.6,
            error_kw={"ecolor": INK_2, "elinewidth": 0.8, "capsize": 0})
    for y, v in enumerate(g["importance_mean"]):
        a2.text(v, y + 0.36, f"{v:.3f}", fontsize=8, color=INK_2, ha="center")
    a2.set_xlabel("Drop in macro-F1 when the whole group is permuted")
    a2.set_title("Feature groups", loc="left")
    a2.grid(axis="y", visible=False)
    fig.tight_layout()
    _save(fig, path)


def plot_feature_distributions(df: pd.DataFrame, features: list[str], path: Path) -> None:
    """Small multiples: per-class distribution (box = IQR, whiskers 5-95 %)
    of selected features.  Log-scale where the feature spans decades."""
    ncol = 4
    nrow = int(np.ceil(len(features) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 3.1 * nrow))
    for ax, feat in zip(axes.ravel(), features):
        data = [df.loc[df.label == c, feat].values for c in CLASSES]
        positive = all((x > 0).all() for x in data)
        spans = positive and max(np.percentile(x, 95) for x in data) / max(min(np.percentile(x, 5) for x in data), 1e-9) > 50
        bp = ax.boxplot(data, whis=(5, 95), showfliers=False, patch_artist=True, widths=0.62,
                        medianprops={"color": INK, "linewidth": 1.2},
                        whiskerprops={"color": INK_2, "linewidth": 0.9}, capprops={"color": INK_2, "linewidth": 0.9})
        for patch, c in zip(bp["boxes"], CLASSES):
            patch.set(facecolor=CLASS_COLORS[c], edgecolor=SURFACE, linewidth=1.5, alpha=0.9)
        if spans:
            ax.set_yscale("log")
        ax.set_xticks(range(1, len(CLASSES) + 1), [pretty(c) for c in CLASSES], rotation=35, ha="right")
        ax.set_title(feat + (" (log)" if spans else ""), loc="left", fontsize=10)
        ax.grid(axis="x", visible=False)
    for ax in axes.ravel()[len(features):]:
        ax.set_visible(False)
    fig.suptitle("Per-class feature distributions (box: IQR, whiskers: 5th-95th pct.)", x=0.01, ha="left",
                 fontweight="bold", color=INK)
    fig.tight_layout()
    _save(fig, path)


def plot_example_traces(npz_path: Path, path: Path) -> None:
    """One example flow per class: packet size vs. time, upstream drawn
    below the axis (negative) and downstream above it."""
    z = np.load(npz_path)
    keys = sorted({k.rsplit("__", 1)[0] for k in z.files})
    first = {}
    for k in keys:
        lab = k.split("__")[0]
        first.setdefault(lab, k)
    fig, axes = plt.subplots(3, 2, figsize=(12, 8.4), sharey=True)
    for ax, lab in zip(axes.ravel(), CLASSES):
        k = first[lab]
        t, s, d = z[k + "__t"], z[k + "__s"], z[k + "__d"]
        sign = np.where(d == 1, 1, -1)
        for dirn, col, name in ((1, SERIES[0], "downstream"), (0, SERIES[1], "upstream")):
            m = d == dirn
            ax.scatter(t[m], (s * sign)[m], s=3, color=col, linewidths=0, alpha=0.6, label=name, rasterized=True)
        ax.axhline(0, color=INK_2, linewidth=0.6)
        sub = k.split("__")[2]
        ax.set_title(f"{pretty(lab)}  ·  {sub}  ·  {t.size:,} packets", loc="left", fontsize=10)
        ax.set_xlim(0, 60)
        ax.set_ylim(-1600, 1600)
    for ax in axes[:, 0]:
        ax.set_ylabel("Packet size (bytes)\n← up   down →")
    for ax in axes[-1]:
        ax.set_xlabel("Time since first packet (s)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", markerscale=4, ncol=2)
    fig.suptitle("Example encrypted flows as seen by the observer (metadata only)", x=0.01, ha="left",
                 fontweight="bold", color=INK)
    fig.tight_layout()
    _save(fig, path)


def plot_model_comparison(m: pd.DataFrame, path: Path) -> None:
    """Dot plot: test macro-F1 and 5-fold CV mean ± std for each model."""
    m = m.sort_values("f1_macro")
    y = np.arange(len(m))
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.errorbar(m["cv_f1_macro_mean"], y + 0.12, xerr=m["cv_f1_macro_std"], fmt="o", ms=7, mfc=SURFACE,
                mec=SERIES[1], mew=2, ecolor=SERIES[1], elinewidth=2, capsize=0, label="5-fold CV (train) mean ± std")
    ax.plot(m["f1_macro"], y - 0.12, "o", ms=8, color=SERIES[0], mec=SURFACE, mew=1.5, label="held-out test")
    for yi, v in zip(y, m["f1_macro"]):
        ax.text(v, yi - 0.42, f"{v:.4f}", ha="center", fontsize=8, color=INK_2)
    ax.set_yticks(y, m["model"])
    ax.set_xlabel("Macro-F1")
    ax.set_ylim(-0.7, len(m) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    ax.set_title("Model comparison", loc="left")
    _save(fig, path)


def plot_early(early: pd.DataFrame, per_class: pd.DataFrame, model: str, path: Path) -> None:
    """Macro-F1 / accuracy (left) and per-class F1 (right) vs. #packets seen."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
    x = early["n_packets_num"]
    a1.plot(x, early["f1_macro"], "-o", color=SERIES[0], lw=2, ms=6, mec=SURFACE, label="macro-F1")
    a1.plot(x, early["accuracy"], "-o", color=SERIES[1], lw=2, ms=6, mec=SURFACE, label="accuracy")
    for xi, v in zip(x, early["f1_macro"]):
        a1.text(xi, v - 0.06, f"{v:.2f}", ha="center", fontsize=7.5, color=INK_2)
    a1.legend(loc="lower right")
    a1.set_ylabel("Score (test set)")
    a1.set_title(f"Early classification ({model})", loc="left")
    for c in CLASSES:
        pc = per_class[per_class["class"] == c]
        a2.plot(pc["n_packets_num"], pc["f1"], "-o", lw=2, ms=5, color=CLASS_COLORS[c], mec=SURFACE, label=pretty(c))
    a2.legend(loc="lower right", ncol=2)
    a2.set_title("Per-class F1", loc="left")
    ticks = list(x)
    for ax in (a1, a2):
        ax.set_xscale("log")
        ax.set_xticks(ticks, [str(int(v)) for v in ticks[:-1]] + ["full\nflow"])
        ax.minorticks_off()
        ax.set_xlabel("Packets observed from flow start")
        ax.set_ylim(0, 1.04)
    fig.tight_layout()
    _save(fig, path)


def _grouped_bars(ax, cats, series: dict, fmt="{:.2f}"):
    n = len(series)
    w = 0.8 / n
    x = np.arange(len(cats))
    for i, (name, vals) in enumerate(series.items()):
        xs = x + (i - (n - 1) / 2) * w
        ax.bar(xs, vals, width=w - 0.03, color=SERIES[i], label=name)
        for xi, v in zip(xs, vals):
            ax.text(xi, v + 0.012, fmt.format(v), ha="center", fontsize=7.5, color=INK_2)
    ax.set_xticks(x, cats)
    ax.grid(axis="x", visible=False)


def plot_defenses(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.2))
    cats = [f"{r.defense}\n+{100 * r.byte_overhead:,.0f}% bytes\n+{r.median_flow_mean_delay_ms:,.0f} ms delay"
            if r.defense != "none" else "none" for r in df.itertuples()]
    _grouped_bars(ax, cats, {
        "adaptive attacker (retrained on defended traffic)": df["f1_macro_adaptive"].values,
        "non-adaptive attacker (trained on undefended traffic)": df["f1_macro_nonadaptive"].values,
    })
    ax.axhline(1 / len(CLASSES), color=INK_2, lw=0.8, ls="--")
    ax.text(len(df) - 0.5, 1 / len(CLASSES) + 0.015, "chance", ha="right", fontsize=8, color=INK_2)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Macro-F1 (test set)")
    ax.legend(loc="upper center", ncol=1, bbox_to_anchor=(0.5, -0.2))
    ax.set_title("Countermeasures vs. classifier", loc="left")
    _save(fig, path)


def plot_ablation(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    _grouped_bars(ax, list(df["group"]), {
        "only this group": df["f1_only"].values,
        "all features except this group": df["f1_without"].values,
    })
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Macro-F1 (test set)")
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.12))
    ax.set_title("Feature-group ablation", loc="left")
    _save(fig, path)
