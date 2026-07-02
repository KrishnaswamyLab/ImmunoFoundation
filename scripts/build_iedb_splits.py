"""Build per-split CSVs for IEDB fine-tuning by joining immunostruct splits with merged_af3.csv.

For each immunostruct split file (train/val/test), join on (allele, peptide) against the AF3-folded
CSV and emit a per-split CSV with the structure path attached. Rows whose (allele, peptide) pair
has no AF3 structure are dropped (the script reports the drop count).

Idempotent: re-run after merged_af3.csv grows to pick up newly folded structures.
"""

import os
import sys

import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MERGED_AF3 = os.path.join(REPO, "data", "IEDB", "merged_af3.csv")
SPLITS = {
    "train": os.path.join(REPO, "data", "IEDB", "immunostruct_train_set.txt"),
    "val": os.path.join(REPO, "data", "IEDB", "immunostruct_val_set.txt"),
    "test": os.path.join(REPO, "data", "IEDB", "immunostruct_test_set.txt"),
}
OUT_DIR = os.path.join(REPO, "data", "IEDB", "splits")


def load_split(path):
    df = pd.read_csv(path, sep="\t")
    # train and val have an unnamed leading index column; drop if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    required = {"peptide", "allele", "immunogenicity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}")
    return df


def main():
    if not os.path.exists(MERGED_AF3):
        sys.exit(f"merged_af3.csv not found at {MERGED_AF3}")

    csv = pd.read_csv(MERGED_AF3)
    csv_lookup = {
        (row["allele"], row["mut_pep"]): {"cif_path": row["cif_path"], "stem": row["stem"], "label_csv": int(row["immunogenicity"])}
        for _, row in csv.iterrows()
    }
    print(f"Loaded merged_af3.csv: {len(csv)} rows, {len(csv_lookup)} unique (allele, mut_pep) keys")

    os.makedirs(OUT_DIR, exist_ok=True)

    for name, path in SPLITS.items():
        df = load_split(path)
        rows_in = len(df)
        kept = []
        dropped = 0
        label_disagree = 0
        for src_idx, row in df.iterrows():
            key = (row["allele"], row["peptide"])
            hit = csv_lookup.get(key)
            if hit is None:
                dropped += 1
                continue
            label_split = int(row["immunogenicity"])
            if hit["label_csv"] != label_split:
                label_disagree += 1
                continue
            kept.append({
                "cif_path": hit["cif_path"],
                "allele": row["allele"],
                "peptide": row["peptide"],
                "immunogenicity": label_split,
                "stem": hit["stem"],
                "source_row_idx": int(src_idx),
            })
        if label_disagree:
            sys.exit(f"{name}: {label_disagree} rows have a label disagreement between split and CSV — aborting")

        out = pd.DataFrame(kept, columns=["cif_path", "allele", "peptide", "immunogenicity", "stem", "source_row_idx"])
        out_path = os.path.join(OUT_DIR, f"{name}.csv")
        out.to_csv(out_path, index=False)
        unique_keys = out[["allele", "peptide"]].drop_duplicates().shape[0]
        n_pos = int((out["immunogenicity"] == 1).sum())
        n_neg = int((out["immunogenicity"] == 0).sum())
        print(
            f"{name:5s}: {rows_in:>6} rows in -> {len(out):>6} kept, {dropped:>6} dropped (no AF3 structure)"
            f"   [unique kept: {unique_keys:>6}, pos: {n_pos}, neg: {n_neg}]"
        )
        print(f"          -> {out_path}")


if __name__ == "__main__":
    main()
