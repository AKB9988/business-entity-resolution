"""
model.py
========
Person 4 — model creation, configuration, and save/load.

MODEL SELECTION (checked at import time, in this preferred order)
-------------------------------------------------------------------
1. LightGBM (LGBMClassifier)      <- team's requirements.txt lists this
2. CatBoost (CatBoostClassifier)
3. sklearn HistGradientBoostingClassifier (always available with sklearn,
   a strong gradient-boosting fallback if neither of the above is installed)

IMPORTANT — ENVIRONMENT NOTE:
This sandbox has no network access, so neither lightgbm nor catboost could
be installed here (`pip install lightgbm` fails with no matching
distribution). The smoke test in train.py therefore ran on the sklearn
fallback. On your actual machine, where `pip install -r requirements.txt`
was run for real, this same code will automatically pick LightGBM instead --
nothing needs to change, `get_model()` re-checks what's importable every
time it's called.

Positive class = 1 = TRUE MATCH (candidate pair refers to the same
real-world business). This is enforced by how train.py builds labels from
train_ground_truth.tsv, not by anything in this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import joblib

try:
    from lightgbm import LGBMClassifier
    _HAS_LIGHTGBM = True
except ImportError:
    _HAS_LIGHTGBM = False

try:
    from catboost import CatBoostClassifier
    _HAS_CATBOOST = True
except ImportError:
    _HAS_CATBOOST = False

from sklearn.ensemble import HistGradientBoostingClassifier


@dataclass
class ModelConfig:
    # Reproducibility: fixed seed everywhere a model/CV split needs one.
    random_state: int = 42

    # LightGBM params (used only if lightgbm is importable)
    lgbm_params: dict = field(default_factory=lambda: dict(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        class_weight="balanced",   # candidate pairs are heavily imbalanced (mostly non-matches)
        random_state=42,
        n_jobs=-1,
    ))

    # CatBoost params (used only if catboost is importable and lightgbm isn't)
    catboost_params: dict = field(default_factory=lambda: dict(
        iterations=400,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        auto_class_weights="Balanced",
        random_seed=42,
        verbose=False,
    ))

    # sklearn fallback params (used only if neither of the above is importable)
    sklearn_params: dict = field(default_factory=lambda: dict(
        max_iter=300,
        learning_rate=0.05,
        max_depth=None,
        l2_regularization=1.0,
        random_state=42,
        # HistGradientBoostingClassifier has no built-in class_weight in
        # older sklearn; class imbalance is instead handled via
        # `sample_weight` at fit time in train.py.
    ))


def detected_backend() -> str:
    """Which backend get_model() will actually use right now, in this environment."""
    if _HAS_LIGHTGBM:
        return "lightgbm"
    if _HAS_CATBOOST:
        return "catboost"
    return "sklearn_hgb"


def get_model(config: Optional[ModelConfig] = None):
    """
    Returns (model, backend_name). Picks the best available classifier:
    LightGBM > CatBoost > sklearn HistGradientBoostingClassifier.
    """
    cfg = config or ModelConfig()
    backend = detected_backend()

    if backend == "lightgbm":
        return LGBMClassifier(**cfg.lgbm_params), "lightgbm"
    if backend == "catboost":
        return CatBoostClassifier(**cfg.catboost_params), "catboost"
    return HistGradientBoostingClassifier(**cfg.sklearn_params), "sklearn_hgb"


def save_model(model, path: str) -> None:
    joblib.dump(model, path)


def load_model(path: str):
    return joblib.load(path)
