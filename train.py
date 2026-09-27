"""
train.py
========
Person 4 — training entry point.

Wires together the REAL Person 1/2/3 code (imported, not reimplemented):
    text_preprocessor.preprocess_dataframe   (Person 1)
    blocking.generate_candidates             (Person 2)
    feature_engineering.generate_pair_features (Person 3)

then adds what's actually Person 4's job:
    - turn Person 3's unlabeled feature rows into a labeled training set by
      joining against train_ground_truth.tsv
    - GroupKFold cross-validation, grouped by source1_entity_id (mandatory:
      a Source-1 entity's candidate pairs must never be split across the
      train/val boundary, or the model would implicitly see part of an
      entity's answer during validation)
    - out-of-fold probability generation
    - entity-level macro-F0.5 threshold search (see evaluate.py docstring
      for why this is NOT the same as a plain pairwise F0.5 search)
    - final model fit on all labeled data
    - persist model.pkl + threshold.json

ASSUMPTIONS FLAGGED FOR THE TEAM (no validate_submission.py or ground-truth
header names were available anywhere in the files provided at the time this
was written):
  - train_ground_truth.tsv is read POSITIONALLY: column 0 = the Source-1
    entity id, column 1 = a comma-separated list of its true match ids
    (empty/blank for a singleton). This mirrors the exact format Person 2's
    own `write_candidate_pairs()` uses for candidate_pairs.tsv, and matches
    the challenge walkthrough's description of the ground-truth format.
    Reading positionally (not by a hardcoded header name) means this still
    works even if the real header text differs from what we'd guess.
  - dataset file layout follows verify_real_data.py's own paths:
    dataset/train/train_source{1,2,3}.tsv, dataset/train/train_ground_truth.tsv
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(__file__))
from text_preprocessor import preprocess_dataframe          # Person 1 (real)
from blocking import generate_candidates, BlockingConfig     # Person 2 (real)
from feature_engineering import generate_pair_features       # Person 3 (real)
from model import get_model, save_model, ModelConfig, detected_backend
from evaluate import (
    pairwise_precision_recall_f05,
    evaluate_entity_matches,
    find_best_threshold,
)


@dataclass
class TrainConfig:
    dataset_dir: str = "dataset/train"
    source1_file: str = "train_source1.tsv"
    source2_file: str = "train_source2.tsv"
    source3_file: str = "train_source3.tsv"
    ground_truth_file: str = "train_ground_truth.tsv"

    models_dir: str = "models"
    model_path: str = "models/model.pkl"
    threshold_path: str = "models/threshold.json"

    n_splits: int = 5
    random_state: int = 42
    threshold_grid: tuple = field(default_factory=lambda: tuple(np.round(np.arange(0.05, 0.991, 0.01), 3)))


# =============================================================================
# GROUND TRUTH PARSING
# =============================================================================
def load_ground_truth_map(path: str) -> dict[str, set]:
    """
    Reads train_ground_truth.tsv POSITIONALLY (see module docstring for why)
    into {source1_entity_id: set(true_match_ids)} -- an entity with a blank
    second column becomes an entry mapping to an EMPTY set (a singleton),
    not a missing key, so downstream code can't accidentally confuse "no
    ground truth row" with "confirmed singleton".
    """
    gt = pd.read_csv(path, sep="\t", dtype=str)
    id_col, match_col = gt.columns[0], gt.columns[1]
    ground_truth_map: dict[str, set] = {}
    for s1_id, matches in zip(gt[id_col], gt[match_col]):
        matches = "" if pd.isna(matches) else str(matches)
        ids = {m.strip() for m in matches.split(",") if m.strip()}
        ground_truth_map[s1_id] = ids
    return ground_truth_map


# =============================================================================
# FEATURE / LABEL ASSEMBLY
# =============================================================================
NON_FEATURE_COLS = ("source1_entity_id", "candidate_entity_id")


def build_labeled_dataset(
    s1_raw: pd.DataFrame,
    s2_raw: pd.DataFrame,
    s3_raw: pd.DataFrame,
    ground_truth_map: dict[str, set],
    blocking_config: Optional[BlockingConfig] = None,
) -> pd.DataFrame:
    """
    Runs Person 1 -> Person 2 -> Person 3 (real code) on raw training data,
    then attaches the binary `label` column by checking each
    (source1_entity_id, candidate_entity_id) pair against ground_truth_map.
    label = 1  <=>  candidate_entity_id in ground_truth_map[source1_entity_id]
    """
    s1 = preprocess_dataframe(s1_raw)
    s2 = preprocess_dataframe(s2_raw)
    s3 = preprocess_dataframe(s3_raw)

    candidate_pairs = generate_candidates(s1, s2, s3, blocking_config or BlockingConfig())

    candidate_df = pd.concat([s2, s3], ignore_index=True, sort=False)
    features = generate_pair_features(s1, candidate_df, candidate_pairs)

    if features.empty:
        raise RuntimeError(
            "generate_pair_features produced 0 rows -- candidate generation "
            "found no candidates at all for any Source-1 entity. Check "
            "Person 2's blocking output before proceeding."
        )

    features["label"] = [
        1 if cid in ground_truth_map.get(s1_id, set()) else 0
        for s1_id, cid in zip(features["source1_entity_id"], features["candidate_entity_id"])
    ]
    return features, s1["entity_id"].tolist()


# =============================================================================
# TRAINING PIPELINE
# =============================================================================
def run_group_kfold_cv(features: pd.DataFrame, all_s1_ids: list, config: TrainConfig) -> dict:
    """
    GroupKFold CV grouped by source1_entity_id. Returns fold-wise diagnostics
    plus the pooled out-of-fold probability array (used afterward for the
    single entity-level threshold search over the whole training set).
    """
    feature_cols = [c for c in features.columns if c not in NON_FEATURE_COLS + ("label",)]
    X = features[feature_cols].to_numpy(dtype=float)
    y = features["label"].to_numpy(dtype=int)
    groups = features["source1_entity_id"].to_numpy()

    n_groups = len(set(groups))
    n_splits = min(config.n_splits, n_groups) if n_groups >= 2 else 1
    if n_splits < 2:
        raise RuntimeError(
            f"Only {n_groups} distinct Source-1 groups in the training features -- "
            "GroupKFold needs at least 2 to do anything meaningful. Check the "
            "dataset / candidate generation output."
        )

    gkf = GroupKFold(n_splits=n_splits)
    oof_prob = np.zeros(len(y), dtype=float)
    fold_reports = []

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        model, backend = get_model(ModelConfig(random_state=config.random_state))

        if backend == "sklearn_hgb":
            # LightGBM/CatBoost get class balancing via constructor params
            # (class_weight="balanced" / auto_class_weights="Balanced");
            # HistGradientBoostingClassifier has no such param, so balance
            # via sample_weight instead, computed on the TRAIN fold only.
            y_train = y[train_idx]
            classes, counts = np.unique(y_train, return_counts=True)
            weight_per_class = {c: len(y_train) / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
            sample_weight = np.array([weight_per_class[label] for label in y_train])
            model.fit(X[train_idx], y_train, sample_weight=sample_weight)
        else:
            model.fit(X[train_idx], y[train_idx])

        val_prob = model.predict_proba(X[val_idx])[:, 1]
        oof_prob[val_idx] = val_prob

        # Diagnostic-only fold metrics at a provisional 0.5 cutoff (the real
        # threshold is chosen once, after CV, on the pooled OOF predictions).
        pairwise = pairwise_precision_recall_f05(y[val_idx], (val_prob >= 0.5).astype(int))
        fold_s1_ids = features["source1_entity_id"].to_numpy()[val_idx]
        fold_cand_ids = features["candidate_entity_id"].to_numpy()[val_idx]
        entity_level = evaluate_entity_matches(
            fold_s1_ids, fold_cand_ids, val_prob,
            ground_truth_map=_labels_to_gt_map(features, y),
            threshold=0.5,
            all_source1_ids=set(fold_s1_ids),
        )
        fold_reports.append({
            "fold": fold_idx,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "pairwise_precision@0.5": pairwise["precision"],
            "pairwise_recall@0.5": pairwise["recall"],
            "pairwise_f05@0.5": pairwise["f05"],
            "entity_macro_f05@0.5": entity_level["macro_f05"],
        })

    return {
        "oof_prob": oof_prob,
        "fold_reports": fold_reports,
        "feature_cols": feature_cols,
        "backend": backend,
    }


def _labels_to_gt_map(features: pd.DataFrame, y: np.ndarray) -> dict[str, set]:
    """Rebuild a {source1_id: set(true candidate ids)} map straight from the
    labeled feature table itself (used only for the per-fold diagnostic
    metric, where we want ground truth restricted to what's actually visible
    in that fold, not the global map)."""
    gt: dict[str, set] = {}
    for s1_id, cid, label in zip(features["source1_entity_id"], features["candidate_entity_id"], y):
        gt.setdefault(s1_id, set())
        if label == 1:
            gt[s1_id].add(cid)
    return gt


def train(config: Optional[TrainConfig] = None) -> dict:
    config = config or TrainConfig()

    s1_raw = pd.read_csv(os.path.join(config.dataset_dir, config.source1_file), sep="\t", dtype=str)
    s2_raw = pd.read_csv(os.path.join(config.dataset_dir, config.source2_file), sep="\t", dtype=str)
    s3_raw = pd.read_csv(os.path.join(config.dataset_dir, config.source3_file), sep="\t", dtype=str)
    ground_truth_map = load_ground_truth_map(os.path.join(config.dataset_dir, config.ground_truth_file))

    features, all_s1_ids = build_labeled_dataset(s1_raw, s2_raw, s3_raw, ground_truth_map)

    print(f"[train] labeled feature rows: {len(features)} | positives: {int(features['label'].sum())} "
          f"({features['label'].mean():.4%}) | Source-1 entities with >=1 candidate: "
          f"{features['source1_entity_id'].nunique()} / {len(all_s1_ids)}")

    cv_result = run_group_kfold_cv(features, all_s1_ids, config)
    print(f"[train] backend selected: {cv_result['backend']}")
    for fr in cv_result["fold_reports"]:
        print(f"[train] fold {fr['fold']}: n_val={fr['n_val']} "
              f"pairwise_f05@0.5={fr['pairwise_f05@0.5']:.4f} "
              f"entity_macro_f05@0.5={fr['entity_macro_f05@0.5']:.4f}")

    # Threshold search on POOLED out-of-fold predictions, scored against the
    # FULL training Source-1 id list (all_s1_ids) so that any entity with
    # zero candidate rows at all -- a total blocking miss, never fit into
    # any fold's groups array -- is still counted as "predicted empty" in
    # the macro average rather than silently ignored.
    best = find_best_threshold(
        source1_ids=features["source1_entity_id"],
        candidate_ids=features["candidate_entity_id"],
        y_prob=cv_result["oof_prob"],
        ground_truth_map=ground_truth_map,
        thresholds=np.array(config.threshold_grid),
        all_source1_ids=all_s1_ids,
    )
    print(f"[train] best threshold (entity-level macro F0.5, pooled OOF): "
          f"{best['threshold']:.2f} -> macro_f05={best['macro_f05']:.4f} "
          f"(precision={best['macro_precision']:.4f}, recall={best['macro_recall']:.4f})")

    # Final model: refit on ALL labeled data using the chosen backend.
    final_model, backend = get_model(ModelConfig(random_state=config.random_state))
    feature_cols = cv_result["feature_cols"]
    X_all = features[feature_cols].to_numpy(dtype=float)
    y_all = features["label"].to_numpy(dtype=int)
    if backend == "sklearn_hgb":
        classes, counts = np.unique(y_all, return_counts=True)
        weight_per_class = {c: len(y_all) / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
        sample_weight = np.array([weight_per_class[label] for label in y_all])
        final_model.fit(X_all, y_all, sample_weight=sample_weight)
    else:
        final_model.fit(X_all, y_all)

    os.makedirs(config.models_dir, exist_ok=True)
    save_model(final_model, config.model_path)
    metadata = {
        "backend": backend,
        "feature_columns": feature_cols,   # predict.py MUST use this exact order
        "threshold": best["threshold"],
        "cv_macro_f05": best["macro_f05"],
        "cv_macro_precision": best["macro_precision"],
        "cv_macro_recall": best["macro_recall"],
        "n_folds": len(cv_result["fold_reports"]),
        "fold_reports": cv_result["fold_reports"],
        "random_state": config.random_state,
    }
    with open(config.threshold_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[train] saved model -> {config.model_path}")
    print(f"[train] saved threshold/metadata -> {config.threshold_path}")
    return metadata


if __name__ == "__main__":
    train()
