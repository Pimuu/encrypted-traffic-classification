"""Metric helpers shared by all experiments."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

from .config import CLASSES


def summary_metrics(y_true, y_pred) -> dict[str, float]:
    """Accuracy plus macro- and weighted-averaged precision / recall / F1."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def per_class_metrics(y_true, y_pred, model: str) -> pd.DataFrame:
    p, r, f, n = precision_recall_fscore_support(y_true, y_pred, labels=CLASSES, zero_division=0)
    return pd.DataFrame({"model": model, "class": CLASSES, "precision": p, "recall": r, "f1": f, "support": n})


def confusion(y_true, y_pred) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    return pd.DataFrame(cm, index=pd.Index(CLASSES, name="true"), columns=pd.Index(CLASSES, name="predicted"))


def fit_and_score(model, X_tr, y_tr, X_te, y_te) -> tuple[dict, np.ndarray]:
    """Fit, predict and time a model; returns (metrics, predictions)."""
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = model.predict(X_te)
    t_pred = time.perf_counter() - t0
    m = summary_metrics(y_te, pred)
    m["train_time_s"] = t_fit
    m["predict_us_per_flow"] = 1e6 * t_pred / len(y_te)
    return m, pred
