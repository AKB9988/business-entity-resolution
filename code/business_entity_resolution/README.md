# Business Entity Resolution Pipeline

## Module: Text Preprocessing & Multilingual Normalization (Person 1)

This module provides high-speed, country-agnostic normalization for noisy business entity records across **US, India, and France**.

### Features
1. **Unicode Accent Stripping**: Normalizes French diacritics (`é, è, ê, ç` $\rightarrow$ `e, e, e, c`) using NFKD normalization.
2. **Legal Entity Standardization**: Normalizes variations of company legal suffixes (`Pvt Ltd`, `LLC`, `Corp`, `Inc`, `SARL`, `SAS`, `SA`, `LLP`, etc.).
3. **Address Canonicalization**: Expands common abbreviations (`St` $\rightarrow$ `Street`, `Rd` $\rightarrow$ `Road`, `Opp` $\rightarrow$ `Opposite`, `Nr` $\rightarrow$ `Near`, etc.).
4. **Structured Signal Extraction**: Extracts postal/PIN codes (5/6 digits) and street/building numbers for downstream blocking and similarity features.

---

### Installation

```bash
pip install -r requirements.txt
```

---

### Usage Example

```python
import pandas as pd
from src.text_preprocessor import TextPreprocessor

# 1. Initialize preprocessor
preprocessor = TextPreprocessor()

# 2. Read dataset (must use sep="\t")
df_s1 = pd.read_csv("dataset/train/train_source1.tsv", sep="\t", nrows=100)

# 3. Clean and extract features
df_clean = preprocessor.preprocess_dataframe(df_s1)

# Columns added:
# - clean_name: Normalized name
# - clean_address: Normalized address
# - postal_codes: List of extracted postal/PIN codes
# - address_numbers: List of extracted building/plot numbers
# - name_prefix_3: 3-character prefix for blocking index
# - first_token: Leading token for inverted index
print(df_clean[["entity_id", "clean_name", "clean_address", "name_prefix_3"]].head())
```

---

### Running Verification Script

To verify on actual dataset records:

```bash
python verify_real_data.py
```
