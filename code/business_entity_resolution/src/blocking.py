"""
blocking.py
============
Blocking & Candidate Generation for the Business Entity Resolution Challenge.

Given cleaned Source1 / Source2 / Source3 dataframes (output of
text_preprocessor.preprocess_dataframe), produce a small, high-recall
candidate set per Source1 entity and write it to candidate_pairs.tsv.

Pipeline
--------
1. Country-level partitioning (open-set: whatever country strings appear in
   the data are used as-is — nothing is hardcoded to {US, India}, so a novel
   "France" or any other label just becomes its own partition automatically).
2. Two independent blocking *channels*, computed within each country partition:
     a) Token / prefix inverted index  -> exact-ish, cheap, catches close
        near-duplicates and short names where TF-IDF similarity is noisy.
     b) Character n-gram TF-IDF + cosine top-K -> catches typos, word-order
        swaps, partial-token overlaps that the inverted index channel misses.
3. Union the two channels' candidates per Source1 entity, score everything
   with the TF-IDF cosine similarity (recomputed if a pair only came from
   the inverted-index channel), then prune to the top `max_candidates`
   (default 20, keep inside the 15-25 recommended range) by score.
4. Write candidate_pairs.tsv in the exact required format.

Scaling notes
-------------
- All heavy lifting (TF-IDF fit/transform, cosine similarity) is done PER
  COUNTRY PARTITION, not globally — this keeps each matrix multiplication
  small even if the full dataset is large, and is what makes "country-level
  partitioning" a blocking optimization and not just a filter.
- Candidate generation from Source2 and Source3 is done separately and then
  merged, since the two sources may have very different size/noise profiles.
- Everything here is TF-IDF / inverted-index based — no external services,
  no embeddings requiring internet access, consistent with the challenge's
  fair-play rules.

Usage
-----
    from text_preprocessor import preprocess_dataframe
    from blocking import generate_candidates, write_candidate_pairs

    s1 = preprocess_dataframe(pd.read_csv("dataset/test/test_source1.tsv", sep="\t"))
    s2 = preprocess_dataframe(pd.read_csv("dataset/test/test_source2.tsv", sep="\t"))
    s3 = preprocess_dataframe(pd.read_csv("dataset/test/test_source3.tsv", sep="\t"))

    candidates = generate_candidates(s1, s2, s3, max_candidates=20)
    write_candidate_pairs(candidates, "output/candidate_pairs.tsv", s1_ids=s1["entity_id"])
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
@dataclass
class BlockingConfig:
    id_col: str = "entity_id"
    country_col: str = "country"
    name_col: str = "clean_name"          # produced by text_preprocessor
    tokens_col: str = "name_tokens"        # produced by text_preprocessor
    prefix_len: int = 3
    tfidf_top_k: int = 30                  # per-channel retrieval depth, before pruning
    max_candidates: int = 20               # final cap per Source1 entity (keep in 15-25)
    ngram_range: tuple = (2, 4)            # char n-gram range for TF-IDF
    min_token_overlap: int = 1             # min shared tokens to count as an inverted-index hit


# ---------------------------------------------------------------------------
# CHANNEL 1: TOKEN / PREFIX INVERTED INDEX
# ---------------------------------------------------------------------------
def _build_inverted_index(df: pd.DataFrame, config: BlockingConfig) -> dict[str, set[str]]:
    """
    Build an inverted index: token -> set of entity_ids that contain it.
    Also indexes each name's first `prefix_len` characters as a pseudo-token,
    so short/rare names still get *some* blocking key even with few tokens.
    """
    index: dict[str, set[str]] = defaultdict(set)
    for _, row in df.iterrows():
        eid = row[config.id_col]
        tokens = row[config.tokens_col] if isinstance(row[config.tokens_col], list) else []
        for tok in tokens:
            index[tok].add(eid)
        name = row[config.name_col] or ""
        if len(name) >= config.prefix_len:
            index[f"__prefix__{name[:config.prefix_len]}"].add(eid)
    return index


def _inverted_index_candidates(
    s1_row: pd.Series,
    index: dict[str, set[str]],
    config: BlockingConfig,
) -> set[str]:
    """Return candidate entity_ids for one Source1 row via token/prefix lookup."""
    candidates: set[str] = set()
    tokens = s1_row[config.tokens_col] if isinstance(s1_row[config.tokens_col], list) else []
    hit_counts: dict[str, int] = defaultdict(int)
    for tok in tokens:
        for eid in index.get(tok, ()):
            hit_counts[eid] += 1
    for eid, count in hit_counts.items():
        if count >= config.min_token_overlap:
            candidates.add(eid)

    name = s1_row[config.name_col] or ""
    if len(name) >= config.prefix_len:
        candidates |= index.get(f"__prefix__{name[:config.prefix_len]}", set())
    return candidates


# ---------------------------------------------------------------------------
# CHANNEL 2: CHAR N-GRAM TF-IDF + COSINE TOP-K
# ---------------------------------------------------------------------------
def _tfidf_topk_within_partition(
    s1_part: pd.DataFrame,
    target_part: pd.DataFrame,
    config: BlockingConfig,
) -> dict[str, list[tuple[str, float]]]:
    """
    Fit a char n-gram TF-IDF vectorizer on the union of S1 + target names in
    this country partition, transform both sides, and for every S1 row
    retrieve the top-K most cosine-similar target rows.

    Returns: {s1_entity_id: [(target_entity_id, score), ...]} sorted desc by score.
    """
    if s1_part.empty or target_part.empty:
        return {eid: [] for eid in s1_part[config.id_col]}

    corpus = pd.concat(
        [s1_part[config.name_col], target_part[config.name_col]], ignore_index=True
    ).fillna("")

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=config.ngram_range, min_df=1)
    tfidf = vectorizer.fit_transform(corpus)

    n_s1 = len(s1_part)
    s1_vecs = tfidf[:n_s1]
    target_vecs = tfidf[n_s1:]

    # TF-IDF rows from sklearn are L2-normalized by default -> cosine sim = dot product.
    sim_matrix = s1_vecs @ target_vecs.T  # sparse result, shape (n_s1, n_target)
    sim_matrix = sparse.csr_matrix(sim_matrix)

    target_ids = target_part[config.id_col].to_numpy()
    s1_ids = s1_part[config.id_col].to_numpy()

    results: dict[str, list[tuple[str, float]]] = {}
    top_k = min(config.tfidf_top_k, len(target_ids))
    for row_idx in range(sim_matrix.shape[0]):
        row = sim_matrix.getrow(row_idx)
        if row.nnz == 0:
            results[s1_ids[row_idx]] = []
            continue
        cols = row.indices
        scores = row.data
        if len(cols) > top_k:
            top_pos = np.argpartition(-scores, top_k - 1)[:top_k]
            cols = cols[top_pos]
            scores = scores[top_pos]
        order = np.argsort(-scores)
        results[s1_ids[row_idx]] = [
            (target_ids[cols[i]], float(scores[i])) for i in order if scores[i] > 0
        ]
    return results


# ---------------------------------------------------------------------------
# CANDIDATE AGGREGATION & PRUNING
# ---------------------------------------------------------------------------
def _generate_for_one_target_source(
    s1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    config: BlockingConfig,
) -> dict[str, list[tuple[str, float]]]:
    """
    Run both blocking channels for one target source (Source2 OR Source3),
    partitioned by country, and return a merged/scored candidate list per
    Source1 entity: {s1_entity_id: [(target_entity_id, score), ...]}.
    """
    merged: dict[str, list[tuple[str, float]]] = {eid: [] for eid in s1_df[config.id_col]}

    countries = pd.unique(pd.concat([s1_df[config.country_col], target_df[config.country_col]]))
    for country in countries:
        s1_part = s1_df[s1_df[config.country_col] == country]
        target_part = target_df[target_df[config.country_col] == country]
        if s1_part.empty or target_part.empty:
            continue  # nothing to match against in this partition

        # Channel A: inverted index (cheap, catches near-exact token matches)
        inv_index = _build_inverted_index(target_part, config)
        inv_candidates: dict[str, set[str]] = {}
        for _, s1_row in s1_part.iterrows():
            inv_candidates[s1_row[config.id_col]] = _inverted_index_candidates(s1_row, inv_index, config)

        # Channel B: TF-IDF char n-gram cosine top-K (catches typos / partial overlap)
        tfidf_candidates = _tfidf_topk_within_partition(s1_part, target_part, config)

        # Union the two channels, scoring everything by TF-IDF similarity.
        # For inverted-index-only hits with no TF-IDF score computed (shouldn't
        # normally happen since TF-IDF covers the whole partition), default to
        # a small positive floor score so they aren't dropped by pruning.
        for eid in s1_part[config.id_col]:
            score_map = {t_id: score for t_id, score in tfidf_candidates.get(eid, [])}
            for t_id in inv_candidates.get(eid, set()):
                score_map.setdefault(t_id, 0.01)
            ranked = sorted(score_map.items(), key=lambda x: -x[1])
            merged[eid] = ranked

    return merged


def generate_candidates(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    config: BlockingConfig = BlockingConfig(),
) -> dict[str, list[str]]:
    """
    Main entry point. Produces the final candidate_pairs mapping:
        {source1_entity_id: [candidate_entity_id, ...]}   (S2/S3 IDs, deduped,
                                                             pruned to max_candidates)
    Candidates from Source2 and Source3 are generated independently, merged,
    then re-pruned together so the overall cap (`max_candidates`) is respected
    across both sources combined — not per-source.
    """
    s2_scored = _generate_for_one_target_source(s1_df, s2_df, config)
    s3_scored = _generate_for_one_target_source(s1_df, s3_df, config)

    final: dict[str, list[str]] = {}
    for eid in s1_df[config.id_col]:
        combined = (s2_scored.get(eid, []) + s3_scored.get(eid, []))
        combined.sort(key=lambda x: -x[1])
        # Dedup while preserving best score's position (shouldn't collide
        # across sources since IDs are source-prefixed, but stay safe).
        seen = set()
        pruned: list[str] = []
        for cand_id, _score in combined:
            if cand_id in seen:
                continue
            seen.add(cand_id)
            pruned.append(cand_id)
            if len(pruned) >= config.max_candidates:
                break
        final[eid] = pruned
    return final


# ---------------------------------------------------------------------------
# OUTPUT WRITER
# ---------------------------------------------------------------------------
def write_candidate_pairs(
    candidates: dict[str, list[str]],
    output_path: str,
    s1_ids: Optional[pd.Series] = None,
) -> None:
    """
    Write candidate_pairs.tsv in the required format:
        source1_entity_id \t candidate_entity_ids   (comma-separated, no spaces)
    One row per Source1 entity (empty string when no candidates found).
    If `s1_ids` is given, it fixes the exact row set/order (guarantees every
    Source1 test entity gets a row, even if it had zero candidates).
    """
    ordered_ids = list(s1_ids) if s1_ids is not None else list(candidates.keys())
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for eid in ordered_ids:
            cand_list = candidates.get(eid, [])
            f.write(f"{eid}\t{','.join(cand_list)}\n")


# ---------------------------------------------------------------------------
# SELF-TEST (run directly: python blocking.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from text_preprocessor import preprocess_dataframe

    s1_raw = pd.DataFrame({
        "entity_id": ["S1-00001", "S1-00002", "S1-00003"],
        "business_name": ["Sharma Traders Pvt. Ltd.", "Global Traders Inc", "Lonely Singleton Co"],
        "business_address": ["45 MG Road, Kanpur, 208001", "100 Main St, New York, NY 10001", "9 Nowhere Ave, Nowhere"],
        "country": ["India", "US", "US"],
    })
    s2_raw = pd.DataFrame({
        "entity_id": ["S2-00047", "S2-00048", "S2-00099"],
        "business_name": ["SHARMA TRADERS PRIVATE LIMITED", "Global Traders Incorporated", "Totally Different Biz"],
        "business_address": ["45, M.G. Rd, Kanpur - 208001", "100 Main Street, New York, NY 10001", "1 Random Rd, Elsewhere"],
        "country": ["India", "US", "US"],
    })
    s3_raw = pd.DataFrame({
        "entity_id": ["S3-00812", "S3-00813"],
        "business_name": ["Sharma Traders", "Global & Co Traders"],
        "business_address": ["MG Road Kanpur 208001", "100 Main St New York 10001"],
        "country": ["India", "US"],
    })

    s1 = preprocess_dataframe(s1_raw)
    s2 = preprocess_dataframe(s2_raw)
    s3 = preprocess_dataframe(s3_raw)

    cfg = BlockingConfig(max_candidates=20, tfidf_top_k=30)
    cands = generate_candidates(s1, s2, s3, cfg)
    for s1_id, cand_ids in cands.items():
        print(f"{s1_id} -> {cand_ids}")

    write_candidate_pairs(cands, "candidate_pairs_test.tsv", s1_ids=s1["entity_id"])
    print("\n--- candidate_pairs_test.tsv ---")
    print(open("candidate_pairs_test.tsv").read())
