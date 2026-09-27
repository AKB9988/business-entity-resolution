# Business Entity Resolution — Methodology & Architecture Report

## 1. Executive Summary
This project implements an end-to-end, high-performance Entity Resolution pipeline for the Amazon ML Challenge. The pipeline links noisy, unstandardized business entity records from multiple sources (Source 2 and Source 3) against a deduplicated reference source (Source 1).

The architecture is specifically engineered to optimize the **Macro-averaged $F_{0.5}$ score**, heavily penalizing false merges while rewarding singleton identification and maintaining a compact candidate pool per entity.

---

## 2. Pipeline Overview

The pipeline follows a 4-phase architecture:
```
Raw Multi-Source Data (US, India, France)
                 │
                 ▼
[Phase 1: Multilingual Text Normalizer & Entity Extractor]
                 │
                 ▼
[Phase 2: Scalable Country-Partitioned Blocking] ──► output/candidate_pairs.tsv
                 │
                 ▼
[Phase 3: 23 Pairwise Similarity & Distance Features]
                 │
                 ▼
[Phase 4: Gradient Boosted Trees + F0.5 Thresholding] ──► output/matching_results.tsv
```

---

## 3. Data Preprocessing & Multilingual Normalization (Person 1)

### Country-Agnostic Design
To generalize seamlessly to new countries in the test set (such as **France** alongside **US** and **India**), preprocessing is strictly country-agnostic:
1. **Unicode Accent De-diacritization**:
   Uses `unicodedata.normalize('NFKD', ...)` to convert French accented characters (`é, è, ê, ç, à, ô`) to standard ASCII equivalents.
2. **Legal Entity Suffix Standardization**:
   Maps multi-jurisdictional legal suffixes into canonical forms:
   * **US**: `Corp`, `Inc`, `LLC`, `Holdings` $\rightarrow$ standardized tokens.
   * **India**: `Pvt Ltd`, `Limited`, `LLP`, `Enterprises`, `Services` $\rightarrow$ canonical expansions.
   * **France**: `SARL`, `SAS`, `SA`, `EURL`, `SCI` $\rightarrow$ canonical expansions.
3. **Address Canonicalization**:
   Expands standard road abbreviations (`Rd` $\rightarrow$ `Road`, `St` $\rightarrow$ `Street`, `Ave` $\rightarrow$ `Avenue`, `Opp` $\rightarrow$ `Opposite`, `Nr` $\rightarrow$ `Near`, `Bd` $\rightarrow$ `Boulevard`).
4. **Structured Signal Extraction**:
   * Extracts 5-digit (US/France) and 6-digit (India) postal/PIN codes.
   * Extracts street numbers, building numbers, and unit numbers.

---

## 4. Scalable Candidate Generation & Blocking (Person 2)

Blocking reduces the $O(N \times M)$ pairwise comparison space to a tight candidate set of $\le 15\text{--}20$ candidates per Source 1 entity, directly addressing the competition's candidate size criteria.

### Multi-Channel Strategy:
1. **Strict Country Partitioning**:
   Pairs are only compared within the same country partition (e.g., US $\leftrightarrow$ US, France $\leftrightarrow$ France).
2. **Inverted Indexing**:
   Builds hash tables on 3-character normalized name prefixes and leading name tokens for fast $O(1)$ exact/prefix candidate retrieval.
3. **Sparse Character n-gram TF-IDF & Cosine Retrieval**:
   Fits character 3-gram TF-IDF vectorizers (`analyzer='char_wb', ngram_range=(3,3)`) on normalized names. Computes sparse dot-product cosine similarity matrices to retrieve the top-$K$ nearest neighbors.
4. **Union & Pruning**:
   Unions candidate hits across channels, ranks by similarity score, and truncates to the top $K$ candidates per entity.
5. **Output**: Exports compliant **`candidate_pairs.tsv`**.

---

## 5. Pairwise Feature Engineering (Person 3)

For every generated candidate pair `(Source1, Candidate)`, a dense 23-dimensional feature vector is computed:

| Feature Group | Features Computed | Description |
| :--- | :--- | :--- |
| **String Distances** | `name_levenshtein`, `name_jaro_winkler`, `name_token_sort_ratio`, `name_token_set_ratio` | Measures fuzzy character edits and token transposition similarities. |
| **Token Overlaps** | `name_jaccard`, `address_jaccard` | Word-level intersection-over-union for names and addresses. |
| **Phonetic Matching**| `name_phonetic_match` | Soundex phonetic equivalence check on leading entity tokens. |
| **Postal / Address Matching** | `postal_exact_match`, `building_number_exact_match` | Binary match flags for postal codes and building numbers. |
| **Length & Difference Metrics** | `name_length_ratio`, `address_length_ratio`, `name_token_count_diff`, `address_token_count_diff`, `name_character_length_diff`, `address_character_length_diff` | Captures relative completeness and text length discrepancies. |
| **Missingness Indicators** | Missing name/address/postal indicators for Source 1 and candidates | Binary flags signaling missing metadata. |

---

## 6. Machine Learning Model & Metric Optimization (Person 4)

### Model Architecture
* **Algorithm**: Gradient Boosted Decision Trees (**LightGBM / HistGradientBoosting**).
* **Validation Strategy**: **5-Fold GroupKFold** cross-validation, strictly grouped by `source1_entity_id`. This guarantees zero data leakage across folds.
* **Class Imbalance**: Managed via balanced class weighting (`scale_pos_weight` / balanced sample weights) to reflect the high negative-to-positive candidate ratio.

### Macro $F_{0.5}$ Precision Optimization & Singleton Handling
Because $F_{0.5} = \frac{1.25 \times P \times R}{0.25 P + R}$ weights precision **2× higher** than recall:
1. **High Confidence Threshold**:
   The decision threshold $\tau$ is tuned over a fine grid $[0.05, 0.99]$ on out-of-fold predictions to maximize macro $F_{0.5}$.
2. **Singleton Handling**:
   If no candidate clears threshold $\tau$, an empty prediction list is generated. This earns a full 1.0 score for true singletons while preventing destructive false positive merges.

---

## 7. Submission Verification & Compliance
* Both **`matching_results.tsv`** and **`candidate_pairs.tsv`** are validated using **`utils/validate_submission.py`** to guarantee:
  * 100% test entity coverage.
  * Exact candidate subset condition (`matches` $\subseteq$ `candidates`).
  * 0 self-matches and 0 duplicate IDs.
  * License compliance with MIT/Apache 2.0.
