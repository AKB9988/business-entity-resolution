# Business Entity Resolution Challenge — Submission Documentation

**Team Name:** coders  
**Team Members:**  
1. Akhil Pathak  
2. Anshika Tiwari  
3. Abhishek Bhatt  
4. Arpit Pathak  
**Submission Date:** 27-09-2026  

---

## 1. Executive Summary

The team built an end-to-end, four-phase Entity Resolution pipeline that links noisy, unstandardized business records from Source 2 and Source 3 against the deduplicated Source 1 reference. The pipeline is: country-agnostic text normalization (accent stripping, legal-suffix and address-abbreviation standardization, postal/building-number extraction) → a two-channel blocking stage (token/prefix inverted index + character $n$-gram TF-IDF cosine top-$K$) producing a compact, country-partitioned candidate set per Source-1 entity → 21-feature pairwise similarity computation over those candidates → a gradient-boosted binary classifier whose decision threshold is tuned to directly optimize the challenge's actual scoring rule (entity-level macro $F_{0.5}$, with explicit singleton handling) on out-of-fold predictions from grouped cross-validation.

---

## 2. Pipeline Overview

```
Raw Multi-Source Data (US, India, France)
                 │
                 ▼
[Phase 1: Country-Agnostic Text Normalizer & Field Extractor]      (Person 1)
                 │
                 ▼
[Phase 2: Country-Partitioned, Two-Channel Blocking]                (Person 2)
                 │                                    ──► output/candidate_pairs.tsv
                 ▼
[Phase 3: 21 Pairwise Similarity & Distance Features]                (Person 3)
                 │
                 ▼
[Phase 4: Gradient-Boosted Classifier + Entity-Level F0.5 Tuning]   (Person 4)
                 │                                    ──► output/matching_results.tsv
```

---

## 3. Methodology

### 3.1 Problem Analysis

The following noise patterns are explicitly handled by `src/text_preprocessor.py`:

* **Multilingual / accented text**: French diacritics (`é, è, ê, ç, à, ô`) are stripped via Unicode NFKD normalization, since the test set includes France in addition to US and India records.
* **Legal entity suffix variation**: A suffix map standardizes country-specific variants — US (`Corp`, `Inc`, `LLC`, `Holdings`), India (`Pvt Ltd`, `Limited`, `LLP`, `Enterprises`, `Services`), France (`SARL`, `SAS`, `SA`, `EURL`, `SCI`, `SNC`, `ETS`, `CIE`) — into canonical forms before comparison.
* **Address abbreviation variation**: Common abbreviations are expanded (`St` $\rightarrow$ `Street`, `Rd` $\rightarrow$ `Road`, `Ave` $\rightarrow$ `Avenue`, `Opp` $\rightarrow$ `Opposite`, `Nr` $\rightarrow$ `Near`, `Bd` $\rightarrow$ `Boulevard`, plus other French forms such as `R` $\rightarrow$ `Rue`, `Rte` $\rightarrow$ `Route`, `BP` $\rightarrow$ `Boite Postale`).
* **Postal/PIN code variation**: A single regex captures both 6-digit Indian PIN codes and 5-digit US/France postal codes; street/building/unit numbers are also extracted.
* **Missing fields**: The feature layer explicitly encodes six missing-field indicators (name/address/postal, for both the Source-1 and candidate side).
* **Noisy / near-duplicate names**: `src/blocking.py` targets look-alike businesses and singleton entities with no true match in Source 2/3.

### 3.2 Solution Strategy

* **Approach Type**: Blocking + Classifier (candidate generation followed by supervised pairwise classification) — this is what `train.py` / `predict.py` implement end-to-end.
* **Core Innovation**: The classifier is paired with an evaluation and threshold optimization layer built specifically for the challenge's scoring rule. `evaluate.py` implements an entity-level macro $F_{0.5}$ metric — per-Source-1-entity precision/recall/$F_{0.5}$ with explicit singleton handling (an entity with no true match scores 1.0 if predicted empty, 0.0 if any candidate is merged onto it) — then macro-averages across entities, and searches a 0.05–0.99 threshold grid for the value that maximizes this metric on pooled out-of-fold predictions from GroupKFold cross-validation grouped by `source1_entity_id` (so a given Source-1 entity's candidate pairs never straddle the train/val split).

---

## 4. Candidate Generation (Blocking)

`src/blocking.py`'s `generate_candidates()` (imported by both `train.py` and `predict.py`) reduces the $O(N \times M)$ comparison space to a compact candidate set per Source-1 entity, using two channels, run per country partition (country is treated as an open string set, so an unseen label like "France" becomes its own partition automatically — nothing is hard-coded to {US, India}):

1. **Channel 1 — Token Inverted Index**: Builds an index from discriminatory tokens (with overgrown generic buckets pruned to prevent memory explosion), and looks up a Source-1 record's tokens against it (cheap, catches near-exact token matches).
2. **Channel 2 — Character n-gram TF-IDF + Cosine Top-K**: Precomputes character 3-gram TF-IDF vectorizers per country partition and retrieves the top-$K$ (`tfidf_top_k`, default 30) cosine-similar candidates per Source-1 record (catches typos and partial token overlap that Channel 1 misses).

The two channels' outputs are unioned per Source-1 entity, scored by TF-IDF cosine similarity, and pruned to `max_candidates` — default 20, confirmed directly from `BlockingConfig` in `src/blocking.py` — combining Source-2 and Source-3 candidates under one shared cap.

* **Persisting `candidate_pairs.tsv`**: Handled directly by `write_candidate_pairs()`, writing the mandatory `output/candidate_pairs.tsv` alongside `matching_results.tsv`.

---

## 5. Matching Model

### 5.1 Features (21 total)

Computed by `src/feature_engineering.py`'s `compute_pair_features()`, and confirmed as the exact set the saved model was trained on via `threshold.json`'s `feature_columns`:

| Feature Group | Features | Description |
| :--- | :--- | :--- |
| **String distances** | `name_levenshtein`, `name_jaro_winkler`, `name_token_sort_ratio`, `name_token_set_ratio` | Fuzzy character-edit and token-transposition similarity |
| **Token overlap** | `name_jaccard`, `address_jaccard` | Word-level intersection-over-union for names and addresses |
| **Phonetic matching** | `name_phonetic_match` | Soundex code match on the leading name token |
| **Postal / building match** | `postal_exact_match`, `building_number_exact_match` | Binary exact-match flags |
| **Length & difference** | `name_length_ratio`, `address_length_ratio`, `name_token_count_diff`, `address_token_count_diff`, `name_character_length_diff`, `address_character_length_diff` | Relative completeness and length discrepancies |
| **Missingness indicators** | `source1_name_missing`, `candidate_name_missing`, `source1_address_missing`, `candidate_address_missing`, `source1_postal_missing`, `candidate_postal_missing` | Binary flags for absent fields on either side of the pair |

### 5.2 Model Architecture

`model.py` dynamically selects between LightGBM, CatBoost, and a scikit-learn `HistGradientBoostingClassifier` fallback, in that preference order, checked at import time — the production environment (with `lightgbm>=4.0.0` installed per `requirements.txt`) selects LightGBM automatically. Class imbalance is handled via balanced sample weights.

* **Validation Strategy**: 5-fold GroupKFold cross-validation, strictly grouped by `source1_entity_id`, so a Source-1 entity's rows never straddle the train/val split within a fold (zero group leakage).

### 5.3 Threshold Selection

`train.py` runs the 5-fold GroupKFold CV to produce pooled out-of-fold prediction probabilities; `evaluate.py`'s `find_best_threshold()` sweeps thresholds from 0.05 to 0.99 (step 0.01) and selects the value that maximizes the entity-level macro $F_{0.5}$ metric, computed against the full training ground-truth map — including Source-1 entities with zero candidate rows, counted as "predicted empty" rather than dropped. The final model is refit on all labeled data.

---

## 6. Results & Validation

* **Selected Optimal Probability Threshold**: `0.41`
* **Cross-Validation Macro $F_{0.5}$**: `~0.969` (across 5 GroupKFold splits)
* **Official Validator Status**: **`PASS`** (verified locally via `utils/validate_submission.py`)
  * Loaded **1,732,544** test Source 1 entities from `dataset/test/test_source1.tsv`
  * Loaded **4,887,273** target entities from `dataset/test/test_source2.tsv`
  * Loaded **5,082,316** target entities from `dataset/test/test_source3.tsv`
  * 100% test-entity coverage verified (every Source-1 test entity has exactly one row)
  * Candidate-subset condition verified (every matched ID appears in `candidate_pairs.tsv`)
  * Zero self-matches, zero duplicate entity IDs

---

## 7. Submission Verification & Compliance

Both `matching_results.tsv` and `candidate_pairs.tsv` are validated through `utils/validate_submission.py`:

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

* **License Compliance**: Final model uses MIT/Apache-2.0 open-source libraries with $\le 8\text{B}$ parameters.
* **Fair Play Compliance**: Strictly zero external database lookups or web APIs used.

---

## 8. Conclusion

The pipeline chains Person 1 (normalization), Person 2 (blocking), and Person 3 (pairwise feature engineering) modules into a Person 4 training/evaluation/prediction layer that targets the challenge's actual entity-level macro $F_{0.5}$ metric rather than a generic pairwise score. The solution satisfies all schema, singleton, and candidate subset constraints, and is fully packaged for leaderboard evaluation.

---

## Appendix

### A. Code Artefacts
* **Preprocessing entry point**: `code/business_entity_resolution/src/text_preprocessor.py` (`TextPreprocessor` class / `preprocess_dataframe()` wrapper).
* **Blocking entry point**: `code/business_entity_resolution/src/blocking.py` (`generate_candidates()`, `write_candidate_pairs()`, `BlockingConfig`, `TargetIndex`).
* **Feature engineering entry point**: `code/business_entity_resolution/src/feature_engineering.py` (`generate_pair_features()`, `compute_pair_features()`).
* **Training entry point**: `code/business_entity_resolution/src/train.py`. Reads `dataset/train/train_source{1,2,3}.tsv` and `dataset/train/train_ground_truth.tsv`; writes `models/model.pkl` and `models/threshold.json`.
* **Prediction entry point**: `code/business_entity_resolution/src/predict.py`. Reads `dataset/test/test_source{1,2,3}.tsv`; writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
* **Validation entry point**: `utils/validate_submission.py`.

### B. Additional Results Summary

| Metric | Value | Source |
| :--- | :--- | :--- |
| **Number of features** | 21 | `src/feature_engineering.py` / `threshold.json:feature_columns` |
| **Max candidates per Source-1 entity** | 20 (default) | `src/blocking.py:BlockingConfig` |
| **Model backend trained** | `sklearn_hgb` (fallback to LightGBM in production) | `threshold.json:backend` |
| **CV folds** | 5 (GroupKFold) | `threshold.json:n_folds`, `train.py` |
| **Selected probability threshold** | 0.41 | `threshold.json:threshold` |
| **Random seed** | 42 | `threshold.json:random_state` |
| **Source-1 test entities** | 1,732,544 | `dataset/test/test_source1.tsv` |
| **Official validator status** | `PASS` (0 errors) | `utils/validate_submission.py` |
