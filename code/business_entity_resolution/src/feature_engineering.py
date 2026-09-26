# Person 3 - Feature Engineering & Distance Metrics

"""
Computes pairwise similarity features between Source 1
and candidate entity records.
"""

# Person 3 - Feature Engineering & Distance Metrics

from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler


def levenshtein_ratio(text1, text2):
    """
    Levenshtein similarity ratio.
    Returns a value between 0 and 100.
    """
    return fuzz.ratio(text1, text2)


def jaro_winkler(text1, text2):
    """
    Jaro-Winkler similarity.
    Returns a value between 0 and 100.
    """
    return JaroWinkler.normalized_similarity(text1, text2) * 100


def token_sort_ratio(text1, text2):
    """
    Token Sort Ratio.
    Returns a value between 0 and 100.
    """
    return fuzz.token_sort_ratio(text1, text2)


def token_set_ratio(text1, text2):
    """
    Token Set Ratio.
    Returns a value between 0 and 100.
    """
    return fuzz.token_set_ratio(text1, text2)

def jaccard_similarity(tokens1, tokens2):
    """
    Calculate Jaccard similarity between two token collections.

    Jaccard = intersection / union

    Returns a value between 0 and 1.
    """
    set1 = set(tokens1)
    set2 = set(tokens2)

    if not set1 and not set2:
        return 1.0

    if not set1 or not set2:
        return 0.0

    return len(set1 & set2) / len(set1 | set2)

def name_jaccard(name1, name2):
    """
    Jaccard similarity between name tokens.
    """
    tokens1 = name1.lower().split()
    tokens2 = name2.lower().split()

    return jaccard_similarity(tokens1, tokens2)


def address_jaccard(address1, address2):
    """
    Jaccard similarity between address tokens.
    """
    tokens1 = address1.lower().split()
    tokens2 = address2.lower().split()

    return jaccard_similarity(tokens1, tokens2)

def soundex(word):
    """
    Generate the Soundex code for a word.

    Returns a four-character Soundex code.
    """
    if not word:
        return ""

    word = ''.join(ch for ch in word.upper() if ch.isalpha())

    if not word:
        return ""

    first_letter = word[0]

    mapping = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2',
        'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6'
    }

    encoded = []
    previous_code = mapping.get(first_letter, '')

    for char in word[1:]:
        code = mapping.get(char, '')

        if code:
            if code != previous_code:
                encoded.append(code)
            previous_code = code
        else:
            previous_code = ''

    return (first_letter + ''.join(encoded) + '000')[:4]

def phonetic_match(text1, text2):
    """
    Compare the Soundex codes of the first name tokens.

    Returns:
        1.0 if phonetic codes match
        0.0 otherwise
    """
    tokens1 = text1.split()
    tokens2 = text2.split()

    if not tokens1 or not tokens2:
        return 0.0

    return float(soundex(tokens1[0]) == soundex(tokens2[0]))

# ============================================================
# STEP 4: POSTAL CODE & BUILDING NUMBER FEATURES
# ============================================================

def exact_postal_match(postal1, postal2):
    """
    Check whether two postal/PIN codes match exactly.

    Returns:
        1.0 if they match
        0.0 otherwise
    """
    if not postal1 or not postal2:
        return 0.0

    return float(str(postal1).strip() == str(postal2).strip())


def exact_building_number_match(numbers1, numbers2):
    """
    Check whether two records contain at least one
    matching building/street number.

    Returns:
        1.0 if any number matches
        0.0 otherwise
    """
    if not numbers1 or not numbers2:
        return 0.0

    set1 = set(str(x).strip() for x in numbers1)
    set2 = set(str(x).strip() for x in numbers2)

    return float(bool(set1 & set2))


# ============================================================
# STEP 5: LENGTH & MISSING-FIELD FEATURES
# ============================================================

def length_ratio(text1, text2):
    """
    Compare the character lengths of two strings.

    Returns a value between 0 and 1.
    1.0 means equal length.
    """
    text1 = str(text1 or "")
    text2 = str(text2 or "")

    if not text1 and not text2:
        return 1.0

    if not text1 or not text2:
        return 0.0

    return min(len(text1), len(text2)) / max(len(text1), len(text2))


def token_count_difference(text1, text2):
    """
    Absolute difference between token counts.
    """
    tokens1 = str(text1 or "").split()
    tokens2 = str(text2 or "").split()

    return abs(len(tokens1) - len(tokens2))


def character_length_difference(text1, text2):
    """
    Absolute difference between character lengths.
    """
    text1 = str(text1 or "")
    text2 = str(text2 or "")

    return abs(len(text1) - len(text2))


def missing_field(text):
    """
    Returns 1.0 if the field is missing/empty,
    otherwise 0.0.
    """
    if text is None:
        return 1.0

    if str(text).strip() == "":
        return 1.0

    return 0.0


# ============================================================
# STEP 6: COMPLETE PAIRWISE FEATURE GENERATOR
# ============================================================

def compute_pair_features(source1, candidate):
    """
    Compute all Person 3 features for one Source 1 /
    candidate entity pair.

    Parameters
    ----------
    source1 : dict
        Preprocessed Source 1 record.

    candidate : dict
        Preprocessed candidate record.

    Returns
    -------
    dict
        Numerical feature dictionary for Person 4.
    """

    name1 = str(source1.get("clean_name", "") or "")
    name2 = str(candidate.get("clean_name", "") or "")

    address1 = str(source1.get("clean_address", "") or "")
    address2 = str(candidate.get("clean_address", "") or "")

    postal1 = source1.get("primary_postal_code", "")
    postal2 = candidate.get("primary_postal_code", "")

    numbers1 = source1.get("address_numbers", [])
    numbers2 = candidate.get("address_numbers", [])

    features = {

        # -------------------------
        # String similarity
        # -------------------------

        "name_levenshtein":
            levenshtein_ratio(name1, name2),

        "name_jaro_winkler":
            jaro_winkler(name1, name2),

        "name_token_sort_ratio":
            token_sort_ratio(name1, name2),

        "name_token_set_ratio":
            token_set_ratio(name1, name2),

        # -------------------------
        # Token similarity
        # -------------------------

        "name_jaccard":
            name_jaccard(name1, name2),

        "address_jaccard":
            address_jaccard(address1, address2),

        # -------------------------
        # Phonetic similarity
        # -------------------------

        "name_phonetic_match":
            phonetic_match(name1, name2),

        # -------------------------
        # Postal / number checks
        # -------------------------

        "postal_exact_match":
            exact_postal_match(postal1, postal2),

        "building_number_exact_match":
            exact_building_number_match(numbers1, numbers2),

        # -------------------------
        # Length features
        # -------------------------

        "name_length_ratio":
            length_ratio(name1, name2),

        "address_length_ratio":
            length_ratio(address1, address2),

        "name_token_count_diff":
            token_count_difference(name1, name2),

        "address_token_count_diff":
            token_count_difference(address1, address2),

        "name_character_length_diff":
            character_length_difference(name1, name2),

        "address_character_length_diff":
            character_length_difference(address1, address2),

        # -------------------------
        # Missing-field indicators
        # -------------------------

        "source1_name_missing":
            missing_field(name1),

        "candidate_name_missing":
            missing_field(name2),

        "source1_address_missing":
            missing_field(address1),

        "candidate_address_missing":
            missing_field(address2),

        "source1_postal_missing":
            missing_field(postal1),

        "candidate_postal_missing":
            missing_field(postal2),
    }

    return features

def generate_pair_features(source1_df, candidate_df, candidate_pairs):
    """
    Generate feature vectors for all candidate pairs.

    Parameters
    ----------
    source1_df : pandas.DataFrame
        Preprocessed Source 1 records.

    candidate_df : pandas.DataFrame
        Combined preprocessed candidate records from Source 2 and Source 3.

    candidate_pairs : dict
        Mapping:
        source1_entity_id -> list of candidate_entity_ids

    Returns
    -------
    pandas.DataFrame
        One row per candidate pair with similarity features.
    """

    import pandas as pd

    source1_lookup = source1_df.set_index("entity_id").to_dict("index")
    candidate_lookup = candidate_df.set_index("entity_id").to_dict("index")

    rows = []

    for source1_id, candidate_ids in candidate_pairs.items():

        if source1_id not in source1_lookup:
            continue

        source1_record = source1_lookup[source1_id]

        for candidate_id in candidate_ids:

            if candidate_id not in candidate_lookup:
                continue

            candidate_record = candidate_lookup[candidate_id]

            features = compute_pair_features(
                source1_record,
                candidate_record
            )

            features["source1_entity_id"] = source1_id
            features["candidate_entity_id"] = candidate_id

            rows.append(features)

    return pd.DataFrame(rows)