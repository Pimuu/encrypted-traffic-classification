"""Model zoo.  All models are scikit-learn estimators with fixed, documented
hyper-parameters (no test-set tuning) and fixed random seeds."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .config import SEED


def _log_scale():
    """Flow statistics are heavy-tailed and non-negative (bytes span 6 orders
    of magnitude), so linear / neural models get log1p + standardisation.
    Tree models are invariant to monotone transforms and skip this."""
    return [FunctionTransformer(np.log1p, feature_names_out="one-to-one"), StandardScaler()]


def make_models() -> dict:
    """Return fresh, unfitted instances of every compared model."""
    return {
        # linear baseline: multinomial logistic regression
        "LogisticRegression": make_pipeline(*_log_scale(), LogisticRegression(C=1.0, max_iter=3000)),
        # bagged decision trees
        "RandomForest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=1, max_features="sqrt", n_jobs=-1, random_state=SEED),
        # gradient-boosted trees (LightGBM-style histogram implementation)
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1e-3,
            early_stopping=False, random_state=SEED),
        # small feed-forward neural network
        "MLP": make_pipeline(*_log_scale(), MLPClassifier(
            hidden_layer_sizes=(128, 64), alpha=1e-3, learning_rate_init=1e-3, max_iter=600,
            early_stopping=True, n_iter_no_change=20, random_state=SEED)),
    }


def make_model(name: str):
    return make_models()[name]
