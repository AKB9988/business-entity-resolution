"""
Final Submission Generator & Package Builder
Ensures 100% test entity coverage by backfilling singletons for remaining records,
runs full validation, and packages the final submission zip archive.
"""

import os
import sys
import zipfile
import subprocess
import pandas as pd


def finalize():
    print("=" * 80)
    print(" FINALIZING BUSINESS ENTITY RESOLUTION SUBMISSION PACKAGE")
    print("=" * 80)

    base_dir = os.path.abspath(os.path.dirname(__file__))
    test_s1_path = os.path.join(base_dir, "dataset", "test", "test_source1.tsv")
    out_dir = os.path.join(base_dir, "output")
    matching_path = os.path.join(out_dir, "matching_results.tsv")
    candidate_path = os.path.join(out_dir, "candidate_pairs.tsv")

    os.makedirs(out_dir, exist_ok=True)

    # 1. Load all Test Source 1 IDs
    print(f"[1] Loading full test Source 1 entity list from {test_s1_path}...")
    s1_test_df = pd.read_csv(test_s1_path, sep="\t", usecols=["entity_id"], dtype=str)
    all_s1_ids = s1_test_df["entity_id"].tolist()
    total_test_entities = len(all_s1_ids)
    print(f"    [OK] Total Test Source 1 entities: {total_test_entities:,}")

    # 2. Load existing predictions & candidates
    existing_matches = {}
    if os.path.exists(matching_path):
        with open(matching_path, "r", encoding="utf-8") as f:
            for line in f.readlines()[1:]:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) == 2 and parts[0].strip():
                    existing_matches[parts[0].strip()] = parts[1].strip()

    existing_cands = {}
    if os.path.exists(candidate_path):
        with open(candidate_path, "r", encoding="utf-8") as f:
            for line in f.readlines()[1:]:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) == 2 and parts[0].strip():
                    existing_cands[parts[0].strip()] = parts[1].strip()

    print(f"[2] Existing model predictions loaded: {len(existing_matches):,} entities.")

    # 3. Write complete 100% coverage files
    print(f"[3] Writing full {total_test_entities:,} rows to output files...")
    with open(matching_path, "w", encoding="utf-8") as fm:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        for eid in all_s1_ids:
            match_str = existing_matches.get(eid, "")
            fm.write(f"{eid}\t{match_str}\n")

    with open(candidate_path, "w", encoding="utf-8") as fc:
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        for eid in all_s1_ids:
            cand_str = existing_cands.get(eid, "")
            # Ensure candidate pool contains any predicted matches
            match_str = existing_matches.get(eid, "")
            if match_str and not cand_str:
                cand_str = match_str
            fc.write(f"{eid}\t{cand_str}\n")

    print("    [OK] output/matching_results.tsv generated successfully.")
    print("    [OK] output/candidate_pairs.tsv generated successfully.")

    # 4. Run Validator
    print("\n[4] Running official validator check...")
    validator_cmd = [
        sys.executable,
        os.path.join(base_dir, "utils", "validate_submission.py"),
        "--matching", matching_path,
        "--candidate", candidate_path,
        "--test-dir", os.path.join(base_dir, "dataset", "test")
    ]
    res = subprocess.run(validator_cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print(res.stderr)

    # 5. Build submission zip
    zip_name = "team_submission.zip"
    zip_path = os.path.join(base_dir, zip_name)
    print(f"\n[5] Packaging final submission archive -> {zip_name}...")
    
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Output files
        zf.write(matching_path, os.path.join("output", "matching_results.tsv"))
        zf.write(candidate_path, os.path.join("output", "candidate_pairs.tsv"))
        
        # Documentation
        doc_path = os.path.join(base_dir, "Documentation_template.md")
        if os.path.exists(doc_path):
            zf.write(doc_path, "Documentation_template.md")

        # Code package
        code_dir = os.path.join(base_dir, "code", "business_entity_resolution")
        for root, dirs, files in os.walk(code_dir):
            for file in files:
                if file.endswith((".py", ".md", ".txt", ".json")) and not file.endswith((".pyc", ".pkl")):
                    full_p = os.path.join(root, file)
                    rel_p = os.path.relpath(full_p, base_dir)
                    zf.write(full_p, rel_p)

    print(f"    [OK] Submission ZIP created: {zip_path} ({os.path.getsize(zip_path) / 1024:.1f} KB)")
    print("\n" + "=" * 80)
    print("ALL SUBMISSION DELIVERABLES ARE READY FOR UPLOAD!")
    print("=" * 80)


if __name__ == "__main__":
    finalize()
