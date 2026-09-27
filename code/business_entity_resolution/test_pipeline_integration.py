"""
End-to-end Pipeline Integration Test: Person 1 + Person 2 + Person 3
Tests: Preprocessing -> Scalable Blocking -> Pairwise Feature Engineering on real data slices
"""

import os
import sys
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from text_preprocessor import preprocess_dataframe
from blocking import BlockingConfig, generate_candidates, write_candidate_pairs
from feature_engineering import generate_pair_features


def test_integration():
    print("=" * 80)
    print("TESTING INTEGRATION: PERSON 1 + PERSON 2 + PERSON 3")
    print("=" * 80)

    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "..", "..", "dataset", "train")
    s1_path = os.path.join(data_dir, "train_source1.tsv")
    s2_path = os.path.join(data_dir, "train_source2.tsv")
    s3_path = os.path.join(data_dir, "train_source3.tsv")

    print("[1] Loading small slice (100 rows) of real training data...")
    df_s1_raw = pd.read_csv(s1_path, sep="\t", nrows=100)
    df_s2_raw = pd.read_csv(s2_path, sep="\t", nrows=500)
    df_s3_raw = pd.read_csv(s3_path, sep="\t", nrows=500)

    print(f"    Source 1: {len(df_s1_raw)} rows")
    print(f"    Source 2: {len(df_s2_raw)} rows")
    print(f"    Source 3: {len(df_s3_raw)} rows")

    print("\n[2] Person 1: Running Text Preprocessing & Normalization...")
    df_s1_clean = preprocess_dataframe(df_s1_raw)
    df_s2_clean = preprocess_dataframe(df_s2_raw)
    df_s3_clean = preprocess_dataframe(df_s3_raw)
    print("    [OK] Preprocessing successful!")

    print("\n[3] Person 2: Running Blocking & Candidate Generation...")
    cfg = BlockingConfig(max_candidates=15, tfidf_top_k=20)
    candidates = generate_candidates(df_s1_clean, df_s2_clean, df_s3_clean, cfg)
    total_candidates = sum(len(c) for c in candidates.values())
    print(f"    [OK] Generated {total_candidates} total candidate pairs across {len(candidates)} S1 entities")
    print(f"    [OK] Avg candidates per S1: {total_candidates / len(candidates):.2f}")

    print("\n[4] Person 3: Generating Pairwise Similarity Features...")
    df_candidates_all = pd.concat([df_s2_clean, df_s3_clean], ignore_index=True)
    df_features = generate_pair_features(df_s1_clean, df_candidates_all, candidates)
    print(f"    [OK] Feature Matrix Shape: {df_features.shape}")
    print(f"    [OK] Computed {len(df_features.columns)} features per pair:")
    print("       ", list(df_features.columns))

    print("\n[5] Sample Feature Vector Preview:")
    print("-" * 80)
    sample_cols = [
        "source1_entity_id", "candidate_entity_id",
        "name_levenshtein", "name_jaro_winkler", "name_token_sort_ratio",
        "name_jaccard", "address_jaccard", "postal_exact_match", "building_number_exact_match"
    ]
    available_cols = [c for c in sample_cols if c in df_features.columns]
    print(df_features[available_cols].head(5).to_string(index=False))

    print("\n" + "=" * 80)
    print("ALL INTEGRATION TESTS PASSED 100%! READY FOR PERSON 4 MODELING.")
    print("=" * 80)


if __name__ == "__main__":
    test_integration()
