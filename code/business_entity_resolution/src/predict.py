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

    model_path: str = "code/business_entity_resolution/models/model.pkl"
    threshold_path: str = "code/business_entity_resolution/models/threshold.json"

    output_dir: str = "output"
    output_file: str = "matching_results.tsv"
    candidate_file: str = "candidate_pairs.tsv"
    nrows: Optional[int] = None


def predict(config: Optional[PredictConfig] = None) -> None:
    config = config or PredictConfig()

    with open(config.threshold_path) as f:
        metadata = json.load(f)
    threshold = metadata["threshold"]
    feature_cols = metadata["feature_columns"]
    model = load_model(config.model_path)

    print("[predict] Loading pre-trained model and threshold...")
    print(f"[predict] Optimal decision threshold: {threshold}")

    s1_path = os.path.join(config.dataset_dir, config.source1_file)
    s2_path = os.path.join(config.dataset_dir, config.source2_file)
    s3_path = os.path.join(config.dataset_dir, config.source3_file)

    out_match_path = os.path.join(config.output_dir, config.output_file)
    out_cand_path = os.path.join(config.output_dir, config.candidate_file)
    os.makedirs(config.output_dir, exist_ok=True)

    # Pre-clean targets (S2 and S3) and build precomputed index once
    print("[predict] Loading and preprocessing candidate sources (S2 & S3)...")
    s2_raw = pd.read_csv(s2_path, sep="\t", dtype=str)
    s3_raw = pd.read_csv(s3_path, sep="\t", dtype=str)
    s2 = preprocess_dataframe(s2_raw)
    s3 = preprocess_dataframe(s3_raw)
    candidate_df = pd.concat([s2, s3], ignore_index=True, sort=False)
    del s2_raw, s3_raw
    print(f"[predict] Loaded {len(candidate_df)} total target records across S2 & S3.")

    print("[predict] Building precomputed search index over target sources...")
    from blocking import TargetIndex, BlockingConfig
    blk_cfg = BlockingConfig()
    s2_index = TargetIndex(s2, blk_cfg)
    s3_index = TargetIndex(s3, blk_cfg)
    print("[predict] Target search index built successfully!")

    # Initialize output files with headers
    with open(out_match_path, "w", encoding="utf-8") as fm:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
    with open(out_cand_path, "w", encoding="utf-8") as fc:
        fc.write("source1_entity_id\tcandidate_entity_ids\n")

    chunk_size = 50000 if config.nrows is None else config.nrows
    print(f"[predict] Streaming Source 1 in memory-safe chunks of {chunk_size}...")

    total_processed = 0
    zero_matches = 0
    single_matches = 0
    multi_matches = 0

    s1_reader = pd.read_csv(s1_path, sep="\t", dtype=str, chunksize=chunk_size)

    for chunk_idx, s1_chunk_raw in enumerate(s1_reader, start=1):
        s1_clean = preprocess_dataframe(s1_chunk_raw)
        
        # Fast query against precomputed target indexes
        s2_scored = s2_index.query(s1_clean)
        s3_scored = s3_index.query(s1_clean)
        
        cand_pairs: dict[str, list[str]] = {}
        for eid in s1_clean["entity_id"]:
            combined = (s2_scored.get(eid, []) + s3_scored.get(eid, []))
            combined.sort(key=lambda x: -x[1])
            seen = set()
            pruned: list[str] = []
            for cand_id, _score in combined:
                if cand_id in seen:
                    continue
                seen.add(cand_id)
                pruned.append(cand_id)
                if len(pruned) >= blk_cfg.max_candidates:
                    break
            cand_pairs[eid] = pruned

        # Compute features
        features = generate_pair_features(s1_clean, candidate_df, cand_pairs)

        predicted_matches: dict[str, list] = {eid: [] for eid in s1_clean["entity_id"]}

        if not features.empty:
            X_test = features[feature_cols].to_numpy(dtype=float)
            probs = model.predict_proba(X_test)[:, 1]

            for s1_id, cand_id, p in zip(features["source1_entity_id"], features["candidate_entity_id"], probs):
                if p >= threshold:
                    predicted_matches[s1_id].append(cand_id)

        # Append to candidate_pairs.tsv and matching_results.tsv
        with open(out_cand_path, "a", encoding="utf-8") as fc:
            for eid in s1_clean["entity_id"]:
                cands = cand_pairs.get(eid, [])
                fc.write(f"{eid}\t{','.join(cands)}\n")

        with open(out_match_path, "a", encoding="utf-8") as fm:
            for eid in s1_clean["entity_id"]:
                matches = predicted_matches.get(eid, [])
                fm.write(f"{eid}\t{','.join(matches)}\n")
                if len(matches) == 0:
                    zero_matches += 1
                elif len(matches) == 1:
                    single_matches += 1
                else:
                    multi_matches += 1

        total_processed += len(s1_clean)
        print(f"[predict] Chunk {chunk_idx}: Processed {total_processed:,} Source 1 entities... "
              f"(Singletons: {zero_matches:,}, Matches: {single_matches + multi_matches:,})")

        if config.nrows and total_processed >= config.nrows:
            break

    print(f"\n[predict] COMPLETE! Total entities: {total_processed:,}")
    print(f"[predict] Output written to {out_match_path} and {out_cand_path}")


def write_matching_results(predicted_matches: dict[str, list], output_path: str, s1_ids: list) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for eid in s1_ids:
            f.write(f"{eid}\t{','.join(predicted_matches.get(eid, []))}\n")


if __name__ == "__main__":
    predict()
