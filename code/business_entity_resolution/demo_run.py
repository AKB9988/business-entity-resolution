"""
Demonstration Script: Running Preprocessor on Realistic Sample Data
Simulates noisy Source 1, Source 2, and Source 3 entity records.
"""

import os
import sys
import pandas as pd

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from text_preprocessor import TextPreprocessor


def main():
    print("=" * 80)
    print(" BUSINESS ENTITY RESOLUTION - DATA PREPROCESSING DEMO")
    print("=" * 80)

    # Simulated realistic noisy business records across 3 sources
    mock_records = [
        # Example 1: India - Source 1 vs Source 2 noisy pair
        {
            "entity_id": "S1-00001",
            "business_name": "Infosys Technologies Ltd.",
            "business_address": "Plot 44, Electronic City, Hosur Rd, Bangalore, 560100",
            "country": "India",
        },
        {
            "entity_id": "S2-00045",
            "business_name": "Infosys Tech Pvt. Ltd.",
            "business_address": "Near Wipro Gate, Electronics City, Hosur Road, Bengaluru - 560100",
            "country": "India",
        },
        # Example 2: US - Source 1 vs Source 3 noisy pair
        {
            "entity_id": "S1-00002",
            "business_name": "JP Morgan Chase & Co., Inc.",
            "business_address": "270 Park Ave., Fl. 38, New York, NY 10017",
            "country": "US",
        },
        {
            "entity_id": "S3-00112",
            "business_name": "JPMorgan Chase Company",
            "business_address": "270 Park Avenue, 38th Floor, NYC 10017",
            "country": "US",
        },
        # Example 3: France (Test set condition) - Accents and abbreviations
        {
            "entity_id": "S1-00003",
            "business_name": "Société d'Électronique et Télécoms S.A.R.L.",
            "business_address": "12, Boulevard Haussmann, BP 45, 75009 Paris",
            "country": "France",
        },
        {
            "entity_id": "S2-00300",
            "business_name": "Societe d'Electronique & Telecoms SAS",
            "business_address": "12 Bd Haussmann, Paris 75009",
            "country": "France",
        },
    ]

    df_raw = pd.DataFrame(mock_records)
    print("\n[1] RAW INPUT RECORDS (Simulated Source 1, 2, 3 Data):")
    print("-" * 80)
    for _, row in df_raw.iterrows():
        print(f"[{row['entity_id']}] ({row['country']})")
        print(f"  Name   : {row['business_name']}")
        print(f"  Address: {row['business_address']}")

    # Run Preprocessor
    preprocessor = TextPreprocessor()
    df_clean = preprocessor.preprocess_dataframe(df_raw)

    print("\n" + "=" * 80)
    print("[2] CLEANED & NORMALIZED OUTPUT (Ready for Blocking & Similarity Scoring):")
    print("-" * 80)
    for _, row in df_clean.iterrows():
        print(f"\nID: {row['entity_id']} | Country: {row['country']}")
        print(f"  Clean Name     : {row['clean_name']}")
        print(f"  Clean Address  : {row['clean_address']}")
        print(f"  Postal Code(s) : {row['postal_codes']}")
        print(f"  Address Numbers: {row['address_numbers']}")
        print(f"  Blocking Keys  : prefix='{row['name_prefix_3']}', token='{row['first_token']}'")

    print("\n" + "=" * 80)
    print("DEMO RUN FINISHED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
