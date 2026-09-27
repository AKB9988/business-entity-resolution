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

The team built an end-to-end, four-phase Entity Resolution pipeline that links noisy, unstandardized business records from Source 2 and Source 3 against the deduplicated Source 1 reference. 

The pipeline consists of:
1. **Country-agnostic text normalization** (accent stripping, legal-suffix and address-abbreviation standardization, postal/building-number extraction).
2. **Two-channel scalable blocking** (token inverted index + character $n$-gram TF-IDF cosine top-$K$) producing a compact, country-partitioned candidate set per Source-1 entity.
3. **21-feature pairwise similarity computation** over those candidates.
4. **Gradient-boosted binary classifier** whose decision threshold is tuned to directly optimize the challenge's actual scoring rule (entity-level macro $F_{0.5}$, with explicit singleton handling) on out-of-fold predictions from grouped cross-validation.

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
* **Address abbreviation variation**: Common abbreviations are expanded (`St` $\rightarrow$ `Street`, `Rd` $\rightarrow$ `Road`, `Ave` $\rightarrow$ `Avenue`, `Opp` $\rightarrow$ `Opposite`, `Nr` $\rightarrow$ `Near`, `Bd` $\rightarrow$ `Boulevard`, plus French forms such as `R` $\rightarrow$ `Rue`, `Rte` $\rightarrow$ `Route`, `BP` $\rightarrow$ `Boite Postale`).
* **Postal/PIN code variation**: A single regex captures both 6-digit Indian PIN codes and 5-digit US/France postal codes; street/building/unit numbers are also extracted.
* **Missing fields**: The feature layer explicitly encodes six missing-field indicators (name/address/postal, for both the Source-1 and candidate side).
* **Noisy / near-duplicate names**: `src/blocking.py` targets look-alike businesses and singleton entities with no true match in Source 2/3.

### 3.2 Solution Strategy
* **Approach Type**: Blocking + Classifier (candidate generation followed by supervised pairwise classification).
* **Core Innovation**: The classifier is paired with an evaluation and threshold optimization layer built specifically for the challenge's scoring rule. `evaluate.py` implements an entity-level macro $F_{0.5}$ metric with explicit singleton handling (an entity with no true match scores 1.0 if predicted empty, 0.0 if any candidate is merged onto it), macro-averages across entities, and searches a 0.05–0.99 threshold grid for the value that maximizes this metric on pooled out-of-fold predictions from GroupKFold cross-validation grouped by `source1_entity_id`.

---

## 4. Candidate Generation (Blocking)

`src/blocking.py` reduces the $O(N \times M)$ comparison space to a compact candidate set per Source-1 entity, using two channels run per country partition:
1. **Channel 1 — Token Inverted Index**: Builds an index from discriminatory tokens (with overgrown generic buckets pruned to prevent memory explosion), enabling fast lookup.
2. **Channel 2 — Character n-gram TF-IDF + Cosine Top-K**: Precomputes character 3-gram TF-IDF representations per country partition and retrieves top-$K$ cosine-similar candidates per Source-1 record.

The two channels' outputs are unioned per Source-1 entity, scored by TF-IDF cosine similarity, and pruned to `max_candidates` (default 20), combining Source-2 and Source-3 candidates under one shared cap.

* **Candidate Persistence**: Handled directly by `write_candidate_pairs()`, writing compliant `output/candidate_pairs.tsv`.

---

## 5. Matching Model

### 5.1 Features (21 total)
Computed by `src/feature_engineering.py`'s `compute_pair_features()`, confirmed as the exact set saved in `threshold.json:feature_columns`:

| Feature Group | Features | Description |
| :--- | :--- | :--- |
| **String distances** | `name_levenshtein`, `name_jaro_winkler`, `name_token_sort_ratio`, `name_token_set_ratio` | Fuzzy character-edit and token-transposition similarity |
| **Token overlap** | `name_jaccard`, `address_jaccard` | Word-level intersection-over-union for names and addresses |
| **Phonetic matching** | `name_phonetic_match` | Soundex code match on the leading name token |
| **Postal / building match** | `postal_exact_match`, `building_number_exact_match` | Binary exact-match flags |
| **Length & difference** | `name_length_ratio`, `address_length_ratio`, `name_token_count_diff`, `address_token_count_diff`, `name_character_length_diff`, `address_character_length_diff` | Relative completeness and length discrepancies |
| **Missingness indicators** | `source1_name_missing`, `candidate_name_missing`, `source1_address_missing`, `candidate_address_missing`, `source1_postal_missing`, `candidate_postal_missing` | Binary flags for absent fields on either side |

### 5.2 Model Architecture
`model.py` dynamically supports **LightGBM**, **CatBoost**, and scikit-learn's **HistGradientBoostingClassifier**. Class imbalance is handled via balanced sample weights.

* **Validation Strategy**: 5-fold GroupKFold cross-validation, strictly grouped by `source1_entity_id`, ensuring zero group leakage across folds.

### 5.3 Threshold Selection
`train.py` runs 5-fold GroupKFold CV to produce out-of-fold prediction probabilities. `evaluate.py` sweeps thresholds from 0.05 to 0.99 (step 0.01) to maximize the entity-level macro $F_{0.5}$ metric against the full ground-truth map. The final model is refit on all labeled data.

---

## 6. Results & Validation

* **Optimal Probability Threshold**: `0.41`
* **Cross-Validation Macro $F_{0.5}$**: `~0.969` (across 5 GroupKFold splits)
* **Official Validator Result**: `PASS` (verified locally via `utils/validate_submission.py`)
  * Loaded **1,732,544** test Source 1 entities from `dataset/test/test_source1.tsv`
  * Loaded **4,887,273** target entities from `dataset/test/test_source2.tsv`
  * Loaded **5,082,316** target entities from `dataset/test/test_source3.tsv`
  * 100% test entity coverage verified
  * Candidate subset condition verified (`matches` $\subseteq$ `candidates`)
  * 0 self-matches, 0 duplicate IDs

---

## 7. Submission Verification & Compliance

Validation is executed using `utils/validate_submission.py`:
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
* **License Compliance**: All libraries and algorithms (LightGBM, Scikit-learn, RapidFuzz) are MIT/Apache-2.0 licensed with $\le 8\text{B}$ parameters.
* **Fair Play Compliance**: Strictly zero external databases, APIs, or lookups used.

---

## 8. Conclusion

The pipeline seamlessly integrates multilingual normalization (Person 1), candidate blocking (Person 2), pairwise similarity feature extraction (Person 3), and a precision-optimized Gradient Boosted Decision Tree model (Person 4). The resulting submission package adheres strictly to all challenge constraints and delivers high-precision entity resolution.

---

## Appendix: Code Artefacts
* **Preprocessing**: `code/business_entity_resolution/src/text_preprocessor.py`
* **Blocking**: `code/business_entity_resolution/src/blocking.py`
* **Feature Engineering**: `code/business_entity_resolution/src/feature_engineering.py`
* **Model & Evaluation**: `code/business_entity_resolution/src/model.py`, `evaluate.py`, `train.py`, `predict.py`
* **Validator**: `utils/validate_submission.py`
* **Final Package**: `team_submission.zip`
