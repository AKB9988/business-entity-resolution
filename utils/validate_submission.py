#!/usr/bin/env python3
"""
Official Submission Validator for Business Entity Resolution Challenge
Runs locally with Python standard library only (no external dependencies).

Checks both output/matching_results.tsv and output/candidate_pairs.tsv against
all official challenge rules.
"""

import argparse
import os
import sys
from typing import Dict, List, Set, Tuple


def read_tsv_rows(path: str, expected_header: Tuple[str, str]) -> Tuple[List[str], Dict[str, List[str]], List[str]]:
    """
    Reads a submission TSV and returns (ordered_s1_ids, {s1_id: [matched_ids]}, errors).
    """
    errors = []
    ordered_s1_ids = []
    data_map = {}

    if not os.path.exists(path):
        return ordered_s1_ids, data_map, [f"File not found: {path}"]

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if not lines:
        return ordered_s1_ids, data_map, [f"File is empty: {path}"]

    header_line = lines[0].rstrip("\r\n")
    header_parts = header_line.split("\t")
    if len(header_parts) != 2 or header_parts[0] != expected_header[0] or header_parts[1] != expected_header[1]:
        errors.append(
            f"Invalid header in {os.path.basename(path)}: Expected '{expected_header[0]}\\t{expected_header[1]}', "
            f"found '{header_line}'"
        )

    seen_s1 = set()
    for line_num, line in enumerate(lines[1:], start=2):
        line = line.rstrip("\r\n")
        parts = line.split("\t")
        if len(parts) != 2:
            errors.append(f"Line {line_num} in {os.path.basename(path)}: Must have exactly 2 tab-separated columns.")
            continue

        s1_id, match_str = parts[0].strip(), parts[1].strip()

        if s1_id in seen_s1:
            errors.append(f"Duplicate source1_entity_id found at line {line_num}: '{s1_id}'")
        seen_s1.add(s1_id)
        ordered_s1_ids.append(s1_id)

        if not match_str:
            data_map[s1_id] = []
        else:
            match_ids = [m.strip() for m in match_str.split(",") if m.strip()]
            data_map[s1_id] = match_ids

    return ordered_s1_ids, data_map, errors


def validate(matching_path: str, candidate_path: str, test_dir: str) -> bool:
    all_issues = []

    print("=" * 80)
    print("RUNNING SUBMISSION VALIDATION CHECKS")
    print("=" * 80)

    # 1. Check Test Source 1 IDs
    test_s1_path = os.path.join(test_dir, "test_source1.tsv")
    test_s2_path = os.path.join(test_dir, "test_source2.tsv")
    test_s3_path = os.path.join(test_dir, "test_source3.tsv")

    if not os.path.exists(test_s1_path):
        print(f"[!] Warning: Test source 1 file not found at '{test_s1_path}'. Skipping exact S1 entity set check.")
        expected_s1_ids = None
    else:
        with open(test_s1_path, "r", encoding="utf-8") as f:
            lines = [l.split("\t")[0].strip() for l in f.readlines()[1:] if l.strip()]
            expected_s1_ids = set(lines)
            print(f"[OK] Loaded {len(expected_s1_ids)} test Source 1 entities from {test_s1_path}")

    # Load valid target IDs (S2 and S3)
    valid_target_ids = set()
    for tpath, prefix in [(test_s2_path, "S2"), (test_s3_path, "S3")]:
        if os.path.exists(tpath):
            with open(tpath, "r", encoding="utf-8") as f:
                ids = {l.split("\t")[0].strip() for l in f.readlines()[1:] if l.strip()}
                valid_target_ids.update(ids)
                print(f"[OK] Loaded {len(ids)} target entities from {tpath}")

    # 2. Validate matching_results.tsv
    matching_s1_ids, matching_data, matching_errors = read_tsv_rows(
        matching_path, ("source1_entity_id", "matched_entity_ids")
    )
    all_issues.extend(matching_errors)

    # 3. Validate candidate_pairs.tsv
    candidate_s1_ids, candidate_data, candidate_errors = read_tsv_rows(
        candidate_path, ("source1_entity_id", "candidate_entity_ids")
    )
    all_issues.extend(candidate_errors)

    if expected_s1_ids is not None:
        matching_s1_set = set(matching_s1_ids)
        candidate_s1_set = set(candidate_s1_ids)

        diff_missing_matching = expected_s1_ids - matching_s1_set
        if diff_missing_matching:
            all_issues.append(f"matching_results.tsv is missing {len(diff_missing_matching)} Source 1 test entities.")

        diff_missing_candidate = expected_s1_ids - candidate_s1_set
        if diff_missing_candidate:
            all_issues.append(f"candidate_pairs.tsv is missing {len(diff_missing_candidate)} Source 1 test entities.")

    # 4. Check Individual Records: IDs must be S2/S3 only, no duplicates, matches subset of candidates
    for s1_id, match_ids in matching_data.items():
        # Duplicate check within row
        if len(match_ids) != len(set(match_ids)):
            all_issues.append(f"matching_results.tsv has duplicate IDs for entity {s1_id}: {match_ids}")

        # Target ID validity check
        for tid in match_ids:
            if not (tid.startswith("S2-") or tid.startswith("S3-")):
                all_issues.append(f"matching_results.tsv contains invalid ID '{tid}' (must start with S2- or S3-).")
            if valid_target_ids and tid not in valid_target_ids:
                all_issues.append(f"matching_results.tsv contains ID '{tid}' which does not exist in test set.")

        # Candidate subset check
        cand_list = candidate_data.get(s1_id, [])
        cand_set = set(cand_list)
        for tid in match_ids:
            if tid not in cand_set:
                all_issues.append(
                    f"Pipeline Warning/Error: Match ID '{tid}' for {s1_id} in matching_results.tsv "
                    f"was NOT present in candidate_pairs.tsv!"
                )

    for s1_id, cand_ids in candidate_data.items():
        if len(cand_ids) != len(set(cand_ids)):
            all_issues.append(f"candidate_pairs.tsv has duplicate IDs for entity {s1_id}: {cand_ids}")
        for tid in cand_ids:
            if not (tid.startswith("S2-") or tid.startswith("S3-")):
                all_issues.append(f"candidate_pairs.tsv contains invalid ID '{tid}' (must start with S2- or S3-).")

    print("\n" + "-" * 80)
    if not all_issues:
        print("RESULT: PASS")
        print("Your submission files satisfy all structural and schema constraints!")
        print("-" * 80)
        return True
    else:
        print(f"RESULT: FAILED ({len(all_issues)} issues found)")
        for i, issue in enumerate(all_issues[:20], start=1):
            print(f"  {i}. {issue}")
        if len(all_issues) > 20:
            print(f"  ... and {len(all_issues) - 20} more issues.")
        print("-" * 80)
        return False


def main():
    parser = argparse.ArgumentParser(description="Submission validator for Business Entity Resolution")
    parser.add_argument("--matching", default="output/matching_results.tsv", help="Path to matching_results.tsv")
    parser.add_argument("--candidate", default="output/candidate_pairs.tsv", help="Path to candidate_pairs.tsv")
    parser.add_argument("--test-dir", default="dataset/test", help="Path to test dataset directory")
    args = parser.parse_args()

    passed = validate(args.matching, args.candidate, args.test_dir)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
