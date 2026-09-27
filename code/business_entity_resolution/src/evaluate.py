"""
evaluate.py
===========
Person 4 — Evaluation utilities: precision/recall/F0.5, threshold search,
and entity-level (macro) scoring with singleton handling.

WHY TWO KINDS OF METRIC LIVE HERE
----------------------------------
There are two different things you can measure, and they give different
answers:

1. "Pairwise" metrics (`pairwise_precision_recall_f05`): treat every
   (source1_id, candidate_id) row as an independent binary classification
   and compute a single precision/recall/F0.5 over all of them. This is the
   standard, boring way to score a classifier and is useful as a sanity
   check / fold-by-fold diagnostic.

2. "Entity-level macro" metrics (`evaluate_entity_matches`): for EACH
   Source-1 entity, compare its *predicted set* of matched candidate ids
   against its *true set*, compute that one entity's own precision/recall/
   F0.5, and then average across entities (macro-average = every entity
   counts equally, regardless of how many candidates it had).

The challenge's own walkthrough materials describe metric #2 as the actual
scoring rule ("Scored on macro F0.5, precision-weighted... Singletons are
Source 1 entities with no match in Source 2 or 3. Predict an empty list for
them and you earn a full 1.0; predict any match and you score 0. A false
merge costs about twice as much as a miss, so when unsure, do not merge.")
No `utils/validate_submission.py` or official evaluator script was available
in any of the files provided to build this against, so metric #2 here is
implemented directly from that description, not invented independently —
if an official evaluator script surfaces later, swap it in and treat it as
the source of truth instead of this implementation.

Metric #2 is what `find_best_threshold` optimizes for by default. Metric #1
is reported alongside it purely as a diagnostic (it's what most people mean
by "the classifier's F0.5", but it is NOT the same number the leaderboard
would show).
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd


# =============================================================================
# CORE F-BETA
# =============================================================================
def calculate_fbeta(precision: float, recall: float, beta: float = 0.5) -> float:
    """
    F_beta = (1 + beta^2) * precision * recall / (beta^2 * precision + recall)

    beta = 0.5 weights precision more heavily than recall (a false merge is
    treated as worse than a missed match) -- this matches the challenge's
    stated "false merge costs ~2x a miss" framing.
    """
    if precision == 0.0 and recall == 0.0:
        return 0.0
    beta_sq = beta ** 2
    denom = (beta_sq * precision) + recall
    if denom == 0.0:
        return 0.0
    return (1 + beta_sq) * precision * recall / denom


def calculate_f05(precision: float, recall: float) -> float:
    """Convenience wrapper: F_beta with beta fixed at 0.5 (this challenge's metric)."""
    return calculate_fbeta(precision, recall, beta=0.5)


# =============================================================================
# PAIRWISE (DIAGNOSTIC) METRICS
# =============================================================================
def pairwise_precision_recall_f05(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Standard row-level precision/recall/F0.5, treating every candidate pair
    as an independent binary prediction. Diagnostic only -- see module
    docstring for why this is NOT the competition's actual scoring rule.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f05 = calculate_f05(precision, recall)
    return {"precision": precision, "recall": recall, "f05": f05, "tp": tp, "fp": fp, "fn": fn}


def evaluate_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Pairwise precision/recall/F0.5 swept across a threshold grid. Diagnostic."""
    if thresholds is None:
        thresholds = np.arange(0.05, 0.991, 0.01)
    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        m = pairwise_precision_recall_f05(y_true, y_pred)
        m["threshold"] = float(t)
        rows.append(m)
    return pd.DataFrame(rows)


# =============================================================================
# ENTITY-LEVEL (OFFICIAL) METRIC
# =============================================================================
def evaluate_entity_matches(
    source1_ids: Iterable[str],
    candidate_ids: Iterable[str],
    y_prob: np.ndarray,
    ground_truth_map: dict[str, set],
    threshold: float,
    all_source1_ids: Optional[Iterable[str]] = None,
) -> dict:
    """
    Apply `threshold` to get a predicted match SET per Source-1 entity, then
    score each entity against its true match set (from `ground_truth_map`,
    which may map an entity to an EMPTY set for a singleton), and
    macro-average across entities.

    Per-entity scoring rule (matches the challenge's stated singleton rule):
      - true=empty,  predicted=empty  -> precision=1, recall=1, F0.5=1.0
      - true=empty,  predicted=non-empty -> precision=0 (false merge) -> F0.5=0
      - true=non-empty, predicted=empty  -> recall=0 (total miss) -> F0.5=0
      - true=non-empty, predicted=non-empty -> standard precision/recall over
        the intersection, then F0.5

    `all_source1_ids`: pass the FULL list of Source-1 ids being scored (not
    just the ones that happen to have candidate rows) so that an entity with
    ZERO candidate pairs at all (e.g. blocking produced nothing for it) is
    still scored as "predicted empty" rather than silently dropped from the
    macro average -- dropping it would understate how costly a blocking
    miss actually is.
    """
    y_prob = np.asarray(y_prob)
    predicted_by_entity: dict[str, set] = {}
    for s1_id, c_id, p in zip(source1_ids, candidate_ids, y_prob):
        if p >= threshold:
            predicted_by_entity.setdefault(s1_id, set()).add(c_id)

    universe = set(all_source1_ids) if all_source1_ids is not None else set(source1_ids)
    universe |= set(ground_truth_map.keys())

    per_entity_scores = {}
    for s1_id in universe:
        predicted = predicted_by_entity.get(s1_id, set())
        true_set = ground_truth_map.get(s1_id, set())

        if not true_set and not predicted:
            f05 = 1.0
            precision = recall = 1.0
        elif not true_set and predicted:
            precision, recall, f05 = 0.0, 0.0, 0.0  # false merge(s) on a true singleton
        elif true_set and not predicted:
            precision, recall, f05 = 0.0, 0.0, 0.0  # total miss
        else:
            tp = len(predicted & true_set)
            precision = tp / len(predicted) if predicted else 0.0
            recall = tp / len(true_set) if true_set else 0.0
            f05 = calculate_f05(precision, recall)

        per_entity_scores[s1_id] = {"precision": precision, "recall": recall, "f05": f05}

    macro_f05 = float(np.mean([v["f05"] for v in per_entity_scores.values()])) if per_entity_scores else 0.0
    macro_precision = float(np.mean([v["precision"] for v in per_entity_scores.values()])) if per_entity_scores else 0.0
    macro_recall = float(np.mean([v["recall"] for v in per_entity_scores.values()])) if per_entity_scores else 0.0

    return {
        "macro_f05": macro_f05,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "n_entities": len(per_entity_scores),
        "per_entity_scores": per_entity_scores,
    }


def find_best_threshold(
    source1_ids: Iterable[str],
    candidate_ids: Iterable[str],
    y_prob: np.ndarray,
    ground_truth_map: dict[str, set],
    thresholds: Optional[np.ndarray] = None,
    all_source1_ids: Optional[Iterable[str]] = None,
) -> dict:
    """
    Sweep `thresholds` (default 0.05 -> 0.99, step 0.01) and return the one
    that maximizes ENTITY-LEVEL macro F0.5 (the challenge's actual metric,
    per the description in the module docstring) -- NOT pairwise F0.5.
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.991, 0.01)

    source1_ids = list(source1_ids)
    candidate_ids = list(candidate_ids)
    y_prob = np.asarray(y_prob)

    best = {"threshold": None, "macro_f05": -1.0}
    curve = []
    for t in thresholds:
        result = evaluate_entity_matches(
            source1_ids, candidate_ids, y_prob, ground_truth_map, float(t), all_source1_ids
        )
        curve.append({"threshold": float(t), "macro_f05": result["macro_f05"],
                      "macro_precision": result["macro_precision"], "macro_recall": result["macro_recall"]})
        if result["macro_f05"] > best["macro_f05"]:
            best = {"threshold": float(t), "macro_f05": result["macro_f05"],
                    "macro_precision": result["macro_precision"], "macro_recall": result["macro_recall"]}

    best["search_curve"] = pd.DataFrame(curve)
    return best
