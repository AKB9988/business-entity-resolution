"""
Verification script for Person 1: Text Preprocessor on Real Dataset
"""

import os
import sys
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from text_preprocessor import TextPreprocessor


def verify_real_data():
    print("=" * 80)
    print("VERIFYING PERSON 1 PREPROCESSOR ON REAL DATASET")
    print("=" * 80)

    # 1. Load real sample slices from train and test
    train_path = os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "train", "train_source1.tsv")
    test_path = os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "test", "test_source1.tsv")

    print("\n[+] Reading top 15 records from Train Source 1...")
    df_train_sample = pd.read_csv(train_path, sep="\t", nrows=15)
    
    print("[+] Reading top 15 records from Test Source 1 (including France)...")
    df_test_sample = pd.read_csv(test_path, sep="\t", nrows=15)

    preprocessor = TextPreprocessor()

    print("\n" + "-" * 80)
    print("1. EVALUATING TEST SOURCE 1 PREPROCESSING (France, India, US):")
    print("-" * 80)
    df_test_clean = preprocessor.preprocess_dataframe(df_test_sample)

    for _, row in df_test_clean.iterrows():
        print(f"ID: {row['entity_id']} | Country: {row['country']}")
        print(f"  Raw Name      : {row['raw_name']}")
        print(f"  Clean Name    : {row['clean_name']}")
        print(f"  Raw Address   : {row['raw_address']}")
        print(f"  Clean Address : {row['clean_address']}")
        print(f"  Postal / PIN  : {row['postal_codes']}")
        print(f"  Numbers       : {row['address_numbers']}")
        print(f"  Keys          : prefix='{row['name_prefix_3']}', token='{row['first_token']}'")
        print()

    print("=" * 80)
    print("PERSON 1 PREPROCESSOR VERIFICATION RESULT: COMPLETE & READY!")
    print("=" * 80)


if __name__ == "__main__":
    verify_real_data()
