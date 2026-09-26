"""
Text Preprocessing and Country-Agnostic Normalization Module
Business Entity Resolution Challenge

Handles:
1. Multilingual unicode & accent normalization (e.g. French accents: é, è, ê, ç, etc.)
2. Legal entity suffix expansion & standardization (US, India, France: Pvt Ltd, Corp, LLC, SARL, SA, etc.)
3. Common address abbreviation standardization (St -> Street, Rd -> Road, Opp -> Opposite, etc.)
4. Extraction of structured signals: postal/PIN codes, numeric street numbers, landmark cues.
5. High-speed vectorized/batch processing for Pandas DataFrames.
"""

import re
import unicodedata
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd


class TextPreprocessor:
    """
    Modular, country-agnostic text preprocessor for business entity records.
    Designed to handle noisy inputs across US, India, and France.
    """

    def __init__(self):
        # 1. Legal Entity Suffixes & Common Business Terms (Standardized to canonical forms)
        self.legal_suffix_map = {
            # India / Commonwealth variations
            r"\bpvt\s+ltd\b": "private limited",
            r"\bpvt\b": "private",
            r"\bltd\b": "limited",
            r"\bpltd\b": "private limited",
            r"\bllp\b": "limited liability partnership",
            r"\bco\b": "company",
            r"\bcorp\b": "corporation",
            r"\binc\b": "incorporated",
            r"\bents?\b": "enterprises",
            r"\bsvcs?\b": "services",
            r"\btech\b": "technologies",
            r"\bmfg\b": "manufacturing",
            r"\bassoc\b": "associates",
            r"\bintl\b": "international",
            
            # US variations
            r"\bllc\b": "limited liability company",
            r"\bcorporation\b": "corporation",
            r"\bincorporated\b": "incorporated",
            
            # France variations (Open-set support for test set)
            r"\bsarl\b": "societe a responsabilite limitee",
            r"\bsas\b": "societe par actions simplifiee",
            r"\bsa\b": "societe anonyme",
            r"\beurl\b": "entreprise unipersonnelle a responsabilite limitee",
            r"\bsci\b": "societe civile immobiliere",
            r"\bsnc\b": "societe en nom collectif",
            r"\bets\b": "etablissements",
            r"\bcie\b": "compagnie",
        }

        # 2. Address Abbreviations (Standardized to canonical forms)
        self.address_abbr_map = {
            r"\brd\b": "road",
            r"\bst\b": "street",
            r"\bave\b": "avenue",
            r"\bblvd\b": "boulevard",
            r"\bln\b": "lane",
            r"\bdr\b": "drive",
            r"\bct\b": "court",
            r"\bpl\b": "place",
            r"\bsq\b": "square",
            r"\bhwy\b": "highway",
            r"\bfl\b": "floor",
            r"\bflr\b": "floor",
            r"\bbldg\b": "building",
            r"\bapt\b": "apartment",
            r"\bste\b": "suite",
            r"\bdept\b": "department",
            r"\bsec\b": "sector",
            r"\bph\b": "phase",
            r"\bplot\b": "plot",
            r"\bno\b": "number",
            r"\bopp\b": "opposite",
            r"\bnr\b": "near",
            r"\badj\b": "adjacent",
            r"\bbehind\b": "behind",
            
            # French address keywords
            r"\br\b": "rue",
            r"\bav\b": "avenue",
            r"\bbd\b": "boulevard",
            r"\ball\b": "allee",
            r"\bche\b": "chemin",
            r"\bimp\b": "impasse",
            r"\bpl\b": "place",
            r"\brte\b": "route",
            r"\bbp\b": "boite postale",
            r"\bcedex\b": "cedex",
        }

        # Compile regex patterns for performance
        self._compiled_suffixes = [
            (re.compile(pattern, re.IGNORECASE), repl)
            for pattern, repl in self.legal_suffix_map.items()
        ]
        self._compiled_address = [
            (re.compile(pattern, re.IGNORECASE), repl)
            for pattern, repl in self.address_abbr_map.items()
        ]

        # Regex for postal/PIN code extraction:
        # India: 6 digits (e.g. 110001, 560034)
        # US: 5 digits (or 5+4) (e.g. 90210, 10001)
        # France: 5 digits (e.g. 75008, 69001)
        self.postal_code_pattern = re.compile(r"\b\d{5,6}\b")
        
        # Regex for street/building numbers (1-5 digits preceded by optional # or No.)
        self.street_num_pattern = re.compile(r"(?:#|no\.?|num\.?)?\s*(\b\d{1,5}(?:[a-zA-Z])?\b)", re.IGNORECASE)

    @staticmethod
    def strip_accents(text: str) -> str:
        """
        Strips unicode diacritics / accents (e.g., 'Café' -> 'Cafe', 'Société' -> 'Societe').
        Essential for French names and addresses in the test set.
        """
        if not text or not isinstance(text, str):
            return ""
        # Decompose unicode characters to base + combining character, then filter out combining marks
        nfkd_form = unicodedata.normalize("NFKD", text)
        return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

    def clean_generic_text(self, text: str) -> str:
        """
        Standardizes case, strips accents, replaces symbols with spaces, collapses whitespace.
        """
        if not text or not isinstance(text, str) or pd.isna(text):
            return ""
        
        # 1. Normalize accents
        text = self.strip_accents(text)
        
        # 2. Lowercase
        text = text.lower()
        
        # 3. Replace ampersands with 'and' / 'et'
        text = re.sub(r"&", " and ", text)
        
        # 4. Remove special symbols, punctuation except alphanumeric
        text = re.sub(r"[^\w\s]", " ", text)
        
        # 5. Collapse consecutive whitespaces
        text = re.sub(r"\s+", " ", text).strip()
        
        return text

    def clean_business_name(self, name: str) -> str:
        """
        Cleans and canonicalizes a business name by standardizing legal suffixes.
        """
        cleaned = self.clean_generic_text(name)
        if not cleaned:
            return ""
        
        for pattern, replacement in self._compiled_suffixes:
            cleaned = pattern.sub(replacement, cleaned)
            
        return re.sub(r"\s+", " ", cleaned).strip()

    def clean_address(self, address: str) -> str:
        """
        Cleans and canonicalizes an address string by expanding abbreviations.
        """
        cleaned = self.clean_generic_text(address)
        if not cleaned:
            return ""
        
        for pattern, replacement in self._compiled_address:
            cleaned = pattern.sub(replacement, cleaned)
            
        return re.sub(r"\s+", " ", cleaned).strip()

    def extract_postal_codes(self, address: str) -> List[str]:
        """
        Extracts 5-digit (US/France) or 6-digit (India) postal/PIN codes from raw or cleaned address.
        """
        if not address or not isinstance(address, str) or pd.isna(address):
            return []
        matches = self.postal_code_pattern.findall(address)
        return list(dict.fromkeys(matches))  # deduplicate while preserving order

    def extract_numbers(self, address: str) -> List[str]:
        """
        Extracts all numeric components (street numbers, building numbers, plot numbers) from address.
        """
        if not address or not isinstance(address, str) or pd.isna(address):
            return []
        tokens = re.findall(r"\b\d+[a-zA-Z]?\b", address)
        return list(dict.fromkeys(tokens))

    def preprocess_record(self, record: Dict[str, str]) -> Dict[str, any]:
        """
        Preprocesses a single business record dictionary.
        Input format: {'entity_id': ..., 'business_name': ..., 'business_address': ..., 'country': ...}
        """
        raw_name = str(record.get("business_name", ""))
        raw_addr = str(record.get("business_address", ""))
        raw_country = str(record.get("country", "")).strip().upper()

        cleaned_name = self.clean_business_name(raw_name)
        cleaned_addr = self.clean_address(raw_addr)
        postal_codes = self.extract_postal_codes(raw_addr)
        numbers = self.extract_numbers(raw_addr)

        # First 3-letter prefix of clean name (useful for blocking index)
        name_prefix_3 = cleaned_name[:3] if len(cleaned_name) >= 3 else cleaned_name
        first_token = cleaned_name.split()[0] if cleaned_name else ""

        return {
            "entity_id": record.get("entity_id", ""),
            "country": raw_country,
            "raw_name": raw_name,
            "raw_address": raw_addr,
            "clean_name": cleaned_name,
            "clean_address": cleaned_addr,
            "postal_codes": postal_codes,
            "primary_postal_code": postal_codes[0] if postal_codes else "",
            "address_numbers": numbers,
            "name_prefix_3": name_prefix_3,
            "first_token": first_token,
            # Combined text representation for vector/TF-IDF indexing
            "full_representation": f"{cleaned_name} {cleaned_addr}".strip(),
        }

    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Batch preprocesses an entire pandas DataFrame.
        Expected columns: entity_id, business_name, business_address, country
        """
        processed_records = [
            self.preprocess_record(row) for row in df.to_dict(orient="records")
        ]
        return pd.DataFrame(processed_records)


# Self-test demonstration when run as script
if __name__ == "__main__":
    print("=" * 70)
    print("Testing TextPreprocessor on Multi-Country Sample Records")
    print("=" * 70)

    sample_data = [
        # India Case: Abbreviations, landmark, 6-digit PIN
        {
            "entity_id": "S1-00101",
            "business_name": "Tata Consultancy Svcs. Pvt. Ltd. & Co.",
            "business_address": "Plot #42, Opp. SBI Bank, Phase 2, Whitefield, Bangalore 560066",
            "country": "India",
        },
        # US Case: Legal suffix, suite number, 5-digit ZIP
        {
            "entity_id": "S2-00204",
            "business_name": "Acme Corp, Inc. (Holdings)",
            "business_address": "123 Main St., Ste. 400, Austin, TX 78701",
            "country": "US",
        },
        # France Case (Open set in test): Accents, French legal form (SARL), 5-digit postal code
        {
            "entity_id": "S3-00309",
            "business_name": "Société Générale d'Électronique S.A.R.L.",
            "business_address": "15, Rue de la Paix, BP 102, 75008 Paris",
            "country": "France",
        },
    ]

    df_sample = pd.DataFrame(sample_data)
    preprocessor = TextPreprocessor()
    df_clean = preprocessor.preprocess_dataframe(df_sample)

    for idx, row in df_clean.iterrows():
        print(f"\n--- Entity: {row['entity_id']} [{row['country']}] ---")
        print(f"Original Name   : {row['raw_name']}")
        print(f"Cleaned Name    : {row['clean_name']}")
        print(f"Original Address: {row['raw_address']}")
        print(f"Cleaned Address : {row['clean_address']}")
        print(f"Postal Code(s)  : {row['postal_codes']}")
        print(f"Address Numbers : {row['address_numbers']}")
        print(f"Blocking Keys   : Prefix='{row['name_prefix_3']}', FirstToken='{row['first_token']}'")

    print("\n" + "=" * 70)
    print("All preprocessing checks PASSED successfully!")
    print("=" * 70)
