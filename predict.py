"""
predict.py
==========
Person 4 — prediction / final submission generation.

Loads the model + threshold saved by train.py, runs the REAL Person 1/2/3
pipeline on the TEST data (no labels available), applies the tuned
probability threshold, and writes matching_results.tsv using the entity-
level singleton logic described in the challenge materials: for each
Source-1 test entity, predict the set of candidates whose probability
clears the threshold; if none clear it, predict an EMPTY set for that
entity rather than forcing a guess (matches "when unsure, do not merge").

OUTPUT SCHEMA — ASSUMPTION FLAGGED FOR THE TEAM
-------------------------------------------------
No utils/validate_submission.py was available anywhere in the files
provided at the time this was written, so the exact required header names
for matching_results.tsv are not confirmed. This writes it in the same
shape Person 2's own `write_candidate_pairs()` uses for candidate_pairs.tsv
(one row per Source-1 entity, tab-separated, comma-joined match list, empty
string for no match) since that's the one format we DO have direct
evidence for from this repo, and the challenge walkthrough describes
train_ground_truth.tsv using this same one-row-per-entity shape. Header
names used here: `source1_entity_id`, `matched_entity_ids`. RENAME THESE
the moment utils/validate_submission.py or an official schema is available
-- do not assume this guess is correct without checking.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from text_preprocessor import preprocess_dataframe          # Person 1 (real)
from blocking import generate_candidates, BlockingConfig     # Person 2 (real)
from feature_engineering import generate_pair_features       # Person 3 (real)
from model import load_model


@dataclass
class PredictConfig:
    dataset_dir: str = "dataset/test"
    source1_file: str = "test_source1.tsv"
    source2_file: str = "test_source2.tsv"
    source3_file: str = "test_source3.tsv"

    model_path: str = "models/model.pkl"
    threshold_path: str = "models/threshold.json"

    output_dir: str = "output"
    output_file: str = "matching_results.tsv"


def predict(config: Optional[PredictConfig] = None) -> pd.DataFrame:
    config = config or PredictConfig()

    with open(config.threshold_path) as f:
        metadata = json.load(f)
    threshold = metadata["threshold"]
    feature_cols = metadata["feature_columns"]
    model = load_model(config.model_path)

    s1_raw = pd.read_csv(os.path.join(config.dataset_dir, config.source1_file), sep="\t", dtype=str)
    s2_raw = pd.read_csv(os.path.join(config.dataset_dir, config.source2_file), sep="\t", dtype=str)
    s3_raw = pd.read_csv(os.path.join(config.dataset_dir, config.source3_file), sep="\t", dtype=str)

    s1 = preprocess_dataframe(s1_raw)
    s2 = preprocess_dataframe(s2_raw)
    s3 = preprocess_dataframe(s3_raw)

    candidate_pairs = generate_candidates(s1, s2, s3, BlockingConfig())
    candidate_df = pd.concat([s2, s3], ignore_index=True, sort=False)
    features = generate_pair_features(s1, candidate_df, candidate_pairs)

    all_s1_ids = s1["entity_id"].tolist()

    predicted_matches: dict[str, list] = {eid: [] for eid in all_s1_ids}

    if not features.empty:
        # IMPORTANT: use the exact feature column order saved by train.py --
        # relying on dict/column order elsewhere would silently misalign
        # features with what the model was actually trained on.
        missing = set(feature_cols) - set(features.columns)
        if missing:
            raise RuntimeError(
                f"Test features are missing columns the model was trained on: {sorted(missing)}. "
                "This means Person 3's feature_engineering.py output changed shape since training -- "
                "retrain, or investigate the mismatch, before predicting."
            )
        X_test = features[feature_cols].to_numpy(dtype=float)
        probs = model.predict_proba(X_test)[:, 1]

        for s1_id, cand_id, p in zip(features["source1_entity_id"], features["candidate_entity_id"], probs):
            if p >= threshold:
                predicted_matches[s1_id].append(cand_id)

    write_matching_results(predicted_matches, os.path.join(config.output_dir, config.output_file), all_s1_ids)

    n_zero = sum(1 for v in predicted_matches.values() if len(v) == 0)
    n_one = sum(1 for v in predicted_matches.values() if len(v) == 1)
    n_multi = sum(1 for v in predicted_matches.values() if len(v) > 1)
    print(f"[predict] Source-1 entities: {len(all_s1_ids)}")
    print(f"[predict] zero-match: {n_zero} | single-match: {n_one} | multi-match: {n_multi}")
    print(f"[predict] wrote {os.path.join(config.output_dir, config.output_file)}")

    return pd.DataFrame(
        [(eid, ",".join(v)) for eid, v in predicted_matches.items()],
        columns=["source1_entity_id", "matched_entity_ids"],
    )


def write_matching_results(predicted_matches: dict[str, list], output_path: str, s1_ids: list) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for eid in s1_ids:
            f.write(f"{eid}\t{','.join(predicted_matches.get(eid, []))}\n")


if __name__ == "__main__":
    predict()
