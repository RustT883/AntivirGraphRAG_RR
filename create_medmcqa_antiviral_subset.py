import pandas as pd
import re
import hashlib
import json
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Set, Optional

# -----------------------------
# 1) Dictionary loading/cleanup
# -----------------------------
def load_antivirals(antivir_file: str) -> List[str]:
    """
    Load antiviral drug dictionary CSV, expecting column 'Drugs'.
    Cleaning: lowercase, strip, deduplicate.
    """
    df = pd.read_csv(antivir_file)
    if "Drugs" not in df.columns:
        raise ValueError(f"Expected column 'Drugs' in {antivir_file}, found: {list(df.columns)}")

    antivirals = (
        df["Drugs"]
        .dropna()
        .astype(str)
        .map(lambda x: x.strip().lower())
        .tolist()
    )
    antivirals = [a for a in antivirals if a]
    return sorted(set(antivirals))


# -----------------------------
# 2) Regex matching
# -----------------------------
def create_regex_pattern(terms: List[str]) -> re.Pattern:
    """
    Build one big case-insensitive regex with "non-word" boundaries to avoid substrings inside tokens.
    Uses longest-first ordering.
    """
    terms = [t for t in terms if isinstance(t, str) and t.strip()]
    if not terms:
        raise ValueError("Empty term list; cannot create regex pattern.")
    terms_sorted = sorted(terms, key=len, reverse=True)
    pattern = r"(?<!\w)(" + "|".join(re.escape(t) for t in terms_sorted) + r")(?!\w)"
    return re.compile(pattern, flags=re.IGNORECASE)


def search_all_columns(df: pd.DataFrame, pattern: re.Pattern, columns_to_search: List[str]) -> pd.DataFrame:
    """
    Search pattern in multiple columns.
    Returns a DataFrame of matched rows with:
      - match_column: where it matched
      - matches: list of matched strings (as returned by findall)
    NOTE: If the same MedMCQA item matches in multiple columns, it will appear multiple times here.
    """
    results = []
    for col in columns_to_search:
        if col not in df.columns:
            continue

        matches = df[col].astype(str).str.findall(pattern)
        mask = matches.apply(lambda x: len(x) > 0 if x is not None else False)
        matched_rows = df[mask].copy()

        if not matched_rows.empty:
            matched_rows["match_column"] = col
            matched_rows["matches"] = matches[mask]
            results.append(matched_rows)

    if results:
        return pd.concat(results, ignore_index=True)

    return pd.DataFrame()


# -----------------------------
# 3) Dedup by QA fingerprint + audit
# -----------------------------
def normalize_text_for_fp(s: str) -> str:
    """
    Aggressive normalization for robust dedupe:
      - lowercase
      - collapse whitespace
      - remove punctuation (keep a-z0-9 and spaces)
    """
    s = "" if s is None else str(s)
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def qa_fingerprint(row) -> str:
    """
    Fingerprint = SHA1(normalized question + sorted normalized options).
    Sorting options makes it robust to option re-ordering.
    """
    q = normalize_text_for_fp(row.get("question", ""))
    opts = [
        normalize_text_for_fp(row.get("opa", "")),
        normalize_text_for_fp(row.get("opb", "")),
        normalize_text_for_fp(row.get("opc", "")),
        normalize_text_for_fp(row.get("opd", "")),
    ]
    opts = sorted([o for o in opts if o])
    key = q + " || " + " | ".join(opts)
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def dedupe_with_audit(df: pd.DataFrame, prefer_nonempty_exp: bool = True) -> pd.DataFrame:
    """
    Deduplicate by qa_fp and keep an audit trail:
      - dup_count_rows: number of matched rows collapsed into this fingerprint
      - dup_count_unique_ids: number of distinct source IDs (if present)
      - dup_ids: list of original IDs (with repeats to reflect row multiplicity)
      - dup_ids_unique: unique list of original IDs
      - all_match_columns: match columns across collapsed rows
      - all_matches: unique matched terms across collapsed rows
    We keep 1 representative row per fingerprint.
    """
    df = df.copy()
    df["qa_fp"] = df.apply(qa_fingerprint, axis=1)

    # try to find an ID column
    id_col = None
    for cand in ["id", "qid", "question_id", "uid"]:
        if cand in df.columns:
            id_col = cand
            break

    # Prefer the representative row with the longest explanation, if present
    if prefer_nonempty_exp and "exp" in df.columns:
        df["_exp_len"] = df["exp"].fillna("").astype(str).map(len)
        df = df.sort_values(by=["qa_fp", "_exp_len"], ascending=[True, False])
    else:
        df = df.sort_values(by=["qa_fp"])

    out_rows = []
    for fp, g in df.groupby("qa_fp", sort=False):
        rep = g.iloc[0].copy()

        rep["dup_count_rows"] = int(len(g))

        if id_col is not None:
            ids = list(map(str, g[id_col].tolist()))
            rep["dup_ids"] = ids
            rep["dup_ids_unique"] = sorted(set(ids))
            rep["dup_count_unique_ids"] = int(len(set(ids)))
        else:
            rep["dup_ids"] = None
            rep["dup_ids_unique"] = None
            rep["dup_count_unique_ids"] = None

        if "match_column" in g.columns:
            rep["all_match_columns"] = sorted(set(map(str, g["match_column"].dropna().tolist())))
        else:
            rep["all_match_columns"] = None

        if "matches" in g.columns:
            all_m = []
            for cell in g["matches"].dropna().tolist():
                if isinstance(cell, list):
                    all_m.extend(cell)
                else:
                    all_m.append(str(cell))
            rep["all_matches"] = sorted(set(map(str, all_m)))
        else:
            rep["all_matches"] = None

        out_rows.append(rep)

    out = pd.DataFrame(out_rows)
    out = out.drop(columns=["_exp_len"], errors="ignore")
    return out


# -----------------------------
# 4) Main (TRAIN, DRUG ONLY)
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description='Extract antiviral drug-related questions from MedMCQA')
    parser.add_argument('--input', type=str, default='./MedMCQA/train.json',
                       help='Input JSON file path (default: ./MedMCQA/train.json)')
    parser.add_argument('--dict', type=str, default='antivir_list.csv',
                       help='Antiviral drug dictionary CSV file (default: antivir_list.csv)')
    parser.add_argument('--output-raw', type=str, default='medmcqa_antiviral_drug_only_train_raw_matches.csv',
                       help='Output CSV for raw matches (default: medmcqa_antiviral_drug_only_train_raw_matches.csv)')
    parser.add_argument('--output-deduped', type=str, default='medmcqa_antiviral_drug_only_train_deduped.csv',
                       help='Output CSV for deduplicated matches (default: medmcqa_antiviral_drug_only_train_deduped.csv)')
    
    args = parser.parse_args()
    
    # Load data - handle both JSON lines and regular JSON
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    # Try to load as JSON lines first, then as regular JSON
    try:
        df = pd.read_json(input_path, lines=True)
    except ValueError:
        try:
            df = pd.read_json(input_path)
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            sys.exit(1)
    
    print(f"There are {len(df)} questions in the input file")

    search_columns = ["question", "exp", "opa", "opb", "opc", "opd"]

    antivirals = load_antivirals(args.dict)
    print(f"Antiviral dictionary size: {len(antivirals)}")
    print(f"First 10 antivirals: {antivirals[:10]}")

    pattern = create_regex_pattern(antivirals)

    drug_matches = search_all_columns(df, pattern, search_columns)
    print(f"Raw matched rows (antiviral drugs, before dedupe): {len(drug_matches)}")

    if len(drug_matches) == 0:
        print("No matches found. Exiting.")
        return pd.DataFrame()

    deduped = dedupe_with_audit(drug_matches, prefer_nonempty_exp=True)
    print(f"Unique questions after dedupe (by QA fingerprint): {len(deduped)}")

    if "cop" in deduped.columns:
        print("\nCorrect option distribution (cop):")
        print(deduped["cop"].value_counts(dropna=False))

    # Improved top duplicates printout
    if "dup_count_rows" in deduped.columns:
        top = deduped.sort_values("dup_count_rows", ascending=False).head(30).copy()

        def _short(x, n=220):
            if x is None:
                return None
            s = str(x)
            return s if len(s) <= n else s[:n] + " ..."

        # Trim long fields for console readability
        for col, n in [
            ("question", 260),
            ("dup_ids", 220),
            ("dup_ids_unique", 220),
            ("all_match_columns", 120),
            ("all_matches", 220),
        ]:
            if col in top.columns:
                top[col] = top[col].map(lambda x, n=n: _short(x, n))

        print("\nTop duplicate clusters (by dup_count_rows):")
        cols_to_show = [
            c for c in [
                "dup_count_rows",
                "dup_count_unique_ids",
                "dup_ids_unique",
                "all_match_columns",
                "all_matches",
                "question",
            ] if c in top.columns
        ]
        print(top[cols_to_show].to_string(index=False))
        
        # Summary statistics
        print(f"\nSummary Statistics:")
        print(f"Total duplicate clusters: {len(deduped)}")
        print(f"Max duplicates in a cluster: {deduped['dup_count_rows'].max()}")
        print(f"Average duplicates per cluster: {deduped['dup_count_rows'].mean():.2f}")
        if 'dup_count_unique_ids' in deduped.columns:
            print(f"Questions with duplicate unique IDs: {(deduped['dup_count_unique_ids'] > 1).sum()}")

    # Save outputs
    drug_matches.to_csv(args.output_raw, index=False)
    deduped.to_csv(args.output_deduped, index=False)
    print(f"\nOutput saved to:")
    print(f"  Raw matches: {args.output_raw}")
    print(f"  Deduplicated: {args.output_deduped}")

    return deduped


if __name__ == "__main__":
    results = main()
