# Business Entity Resolution — Amazon ML Challenge 2026

An end-to-end, high-performance Machine Learning pipeline designed to resolve and link noisy business entity records across multiple independent data sources (Source 2 and Source 3) to a deduplicated reference source (Source 1).

The pipeline is specifically engineered to optimize the **Macro-averaged $F_{0.5}$ score**, heavily penalizing false merges while rewarding singleton identification and maintaining a compact candidate pool per entity.

---

## 🚀 Key Features

1. **Country-Agnostic Multilingual Preprocessing**:
   - De-diacritization for French text (`é, è, ê, ç` $\rightarrow$ `e, e, e, c`) to support unseen test countries (France).
   - Canonical legal suffix standardization (`Pvt Ltd`, `LLC`, `Corp`, `SARL`, `SAS`, `SA`).
   - Address abbreviation expansions (`St`, `Rd`, `Ave`, `Opp`, `Near`, `Bd`).
   - Automated extraction of 5/6-digit postal/PIN codes and street/building numbers.

2. **Scalable Multi-Channel Candidate Generation (Blocking)**:
   - Country-partitioned search space pruning.
   - Character 3-gram TF-IDF cosine similarity nearest-neighbor retrieval.
   - Exact/prefix inverted hash tables.
   - Generates compact candidate sets ($\le 15\text{--}20$ pairs per S1) saved to `output/candidate_pairs.tsv`.

3. **23 Pairwise Similarity & Distance Features**:
   - Rapid string distances: Levenshtein ratio, Jaro-Winkler, Token Sort Ratio, Token Set Ratio (`rapidfuzz`).
   - Word unigram Jaccard similarity and Soundex phonetic matching.
   - Exact postal/PIN code and building number verification flags.
   - Relative length and discrepancy ratios.

4. **Gradient Boosted Tree Matcher + Precision Optimization**:
   - GroupKFold cross-validation grouped by `source1_entity_id` (zero data leakage).
   - Fine-grained threshold grid search optimizing the macro-averaged $F_{0.5}$ metric.
   - Confident singleton prediction (empty match list = full 1.0 score).

---

## 👥 Team Roles & Responsibilities

| Contributor | Role | Core Responsibility |
| :--- | :--- | :--- |
| **Abhishek Kumar Bhatt** | **Person 1** | Text Preprocessing, Multilingual Normalizer & Signal Extraction |
| **Arpit Pathak** | **Person 2** | Scalable Blocking, Candidate Generation & `candidate_pairs.tsv` |
| **Arpit Pathak** | **Person 3** | Pairwise Feature Engineering (23 Distance & Match Metrics) |
| **Anshika Tiwari** | **Person 4** | LightGBM Model Architecture, GroupKFold, $F_{0.5}$ Thresholding & Predict Pipeline |

---

## 📁 Repository Structure

```text
business-entity-resolution/
│
├── dataset/                                   # Raw TSV dataset files
│   ├── train/ (train_source1.tsv, train_source2.tsv, train_source3.tsv, train_ground_truth.tsv)
│   └── test/  (test_source1.tsv, test_source2.tsv, test_source3.tsv)
│
├── output/                                    # Target competition deliverables
│   ├── matching_results.tsv                   # Final matched entity pairs (Leaderboard submission)
│   └── candidate_pairs.tsv                    # Final candidate pool from blocking
│
├── utils/
│   └── validate_submission.py                 # Self-validation & rule checking script
│
├── Documentation_template.md                  # Comprehensive technical methodology report
│
└── code/
    └── business_entity_resolution/
        ├── requirements.txt                   # Pinned library dependencies
        ├── README.md                          # Source code guide
        ├── models/
        │   ├── model.pkl                      # Trained model weights
        │   └── threshold.json                 # Saved optimal F0.5 decision threshold & metadata
        │
        └── src/
            ├── __init__.py
            ├── text_preprocessor.py           # Person 1: Text cleaning & normalization
            ├── blocking.py                    # Person 2: TF-IDF & inverted index blocking
            ├── feature_engineering.py         # Person 3: 23 pairwise similarity features
            ├── model.py                       # Person 4: LightGBM / CatBoost model class
            ├── evaluate.py                    # Person 4: Macro F0.5 evaluator & singleton scoring
            ├── train.py                       # Person 4: End-to-end GroupKFold training pipeline
            └── predict.py                     # Person 4: Full test set inference pipeline
```

---

## 🛠️ Quickstart & Reproduction Guide

### 1. Installation
```bash
pip install -r code/business_entity_resolution/requirements.txt
```

### 2. Train the Model
Runs preprocessing, candidate generation, 23 feature extraction, GroupKFold validation, and searches for the optimal $F_{0.5}$ decision threshold:
```bash
python -c "import sys; sys.path.insert(0, 'code/business_entity_resolution/src'); from train import train; train()"
```

### 3. Generate Test Predictions
Runs inference on `dataset/test/` and writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`:
```bash
python -c "import sys; sys.path.insert(0, 'code/business_entity_resolution/src'); from predict import predict; predict()"
```

### 4. Validate Submission Files
Run the self-validation check to verify format compliance and candidate subset conditions:
```bash
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```
*(Prints `RESULT: PASS` with exit code 0).*

---

## 📦 Final Submission Package Structure

To submit the final zip package:
```bash
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
```
