"""Build per-split CSVs for CEDAR cancer fine-tuning.

Reads ImmunoStruct_CEDAR_data_cancer.csv, locates the matching AlphaFold2 PDB for each
(allele, mut_pep) pair under data/CEDAR_cancer/alphafold2_pdb/, and emits 80/10/10
train/val/test CSVs with the same column schema as scripts/build_iedb_splits.py.

Strategies (via --strategy):
  random         label-stratified random row split (default; output dir: splits_seed{N}/)
  peptide_group  group-split by peptide so no peptide appears in two splits
                 (output dir: splits_peptide_group_seed{N}/)

Output dir is always seed-suffixed so multi-seed evaluation can rebuild each partition
without overwriting the previous one. Use scripts/run_finetune_cedar_cancer_seeds.sh
to sweep seeds end-to-end.

If the target PDB directory is missing or empty, auto-extracts data/alphafold2_pdb_CEDAR_cancer.zip
and data/alphafold2_pdb_CEDAR_wildtype.zip into the matching subdirs of data/CEDAR_cancer/.

Idempotent: re-run any time. Splits are deterministic given --seed.
"""

import argparse
import os
import sys
import zipfile

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV_PATH = os.path.join(REPO, "data", "ImmunoStruct_CEDAR_data_cancer.csv")
ZIP_PATH = os.path.join(REPO, "data", "alphafold2_pdb_CEDAR_cancer.zip")
ZIP_WT_PATH = os.path.join(REPO, "data", "alphafold2_pdb_CEDAR_wildtype.zip")
PDB_DIR = os.path.join(REPO, "data", "CEDAR_cancer", "alphafold2_pdb")
PDB_WT_DIR = os.path.join(REPO, "data", "CEDAR_cancer", "alphafold2_pdb_wildtype")
SPLIT_BASE = os.path.join(REPO, "data", "CEDAR_cancer")
DEFAULT_SEED = 0
VAL_FRAC = 0.10
TEST_FRAC = 0.10


def extract_zip(zip_path, dest_dir):
    have_pdbs = os.path.isdir(dest_dir) and any(
        f.endswith(".pdb") for f in os.listdir(dest_dir)
    )
    if have_pdbs:
        return
    if not os.path.exists(zip_path):
        sys.exit(f"PDB dir empty and zip missing: {zip_path}")
    os.makedirs(dest_dir, exist_ok=True)
    print(f"Extracting {zip_path} -> {dest_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            name = os.path.basename(member.filename)
            if not name.endswith(".pdb"):
                continue
            target = os.path.join(dest_dir, name)
            with zf.open(member) as src, open(target, "wb") as dst:
                dst.write(src.read())
    n = len([f for f in os.listdir(dest_dir) if f.endswith(".pdb")])
    print(f"Extracted {n} PDB files")


def ensure_pdbs_extracted():
    extract_zip(ZIP_PATH, PDB_DIR)
    extract_zip(ZIP_WT_PATH, PDB_WT_DIR)


def make_stem(allele, peptide, label):
    allele_compact = allele.replace("HLA-", "").replace("*", "").replace(":", "")
    return f"{peptide}_{allele_compact}_imm{label}"


def split_random(df, seed):
    labels = df["immunogenicity"].to_numpy()
    train_df, holdout_df = train_test_split(
        df, test_size=VAL_FRAC + TEST_FRAC, stratify=labels, random_state=seed
    )
    val_df, test_df = train_test_split(
        holdout_df,
        test_size=TEST_FRAC / (VAL_FRAC + TEST_FRAC),
        stratify=holdout_df["immunogenicity"].to_numpy(),
        random_state=seed,
    )
    return train_df, val_df, test_df


def split_peptide_group(df, seed):
    """No peptide appears in two splits. Two-step GroupShuffleSplit."""
    groups = df["peptide"].to_numpy()
    gss1 = GroupShuffleSplit(n_splits=1, test_size=VAL_FRAC + TEST_FRAC, random_state=seed)
    train_idx, holdout_idx = next(gss1.split(df, groups=groups))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    holdout_df = df.iloc[holdout_idx].reset_index(drop=True)

    gss2 = GroupShuffleSplit(
        n_splits=1, test_size=TEST_FRAC / (VAL_FRAC + TEST_FRAC), random_state=seed
    )
    val_idx, test_idx = next(gss2.split(holdout_df, groups=holdout_df["peptide"].to_numpy()))
    val_df = holdout_df.iloc[val_idx].reset_index(drop=True)
    test_df = holdout_df.iloc[test_idx].reset_index(drop=True)

    train_pep = set(train_df["peptide"])
    val_pep = set(val_df["peptide"])
    test_pep = set(test_df["peptide"])
    leak = (train_pep & val_pep) | (train_pep & test_pep) | (val_pep & test_pep)
    if leak:
        sys.exit(f"peptide_group: {len(leak)} peptides leaked across splits — bug")
    return train_df, val_df, test_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy",
        choices=["random", "peptide_group"],
        default="random",
        help="random: label-stratified row split. peptide_group: no peptide in two splits.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    if not os.path.exists(CSV_PATH):
        sys.exit(f"CSV not found: {CSV_PATH}")
    ensure_pdbs_extracted()

    csv = pd.read_csv(CSV_PATH)
    required = {"allele", "mut_pep", "wt_pep", "immunogenicity"}
    missing = required - set(csv.columns)
    if missing:
        sys.exit(f"CSV missing required columns: {missing}")
    print(f"Loaded {CSV_PATH}: {len(csv)} rows")

    rows = []
    dropped_mut = 0
    dropped_wt = 0
    for src_idx, row in csv.iterrows():
        allele = row["allele"]
        peptide = row["mut_pep"]
        wt_peptide = row["wt_pep"]
        label = int(row["immunogenicity"])
        pdb_path = os.path.join(PDB_DIR, f"{allele}_{peptide}.pdb")
        wt_pdb_path = os.path.join(PDB_WT_DIR, f"{allele}_{wt_peptide}.pdb")
        if not os.path.isfile(pdb_path):
            dropped_mut += 1
            continue
        if not os.path.isfile(wt_pdb_path):
            dropped_wt += 1
            continue
        rows.append({
            "cif_path": pdb_path,
            "allele": allele,
            "peptide": peptide,
            "immunogenicity": label,
            "stem": make_stem(allele, peptide, label),
            "source_row_idx": int(src_idx),
            "wt_peptide": wt_peptide,
            "wt_cif_path": wt_pdb_path,
        })
    df = pd.DataFrame(
        rows,
        columns=[
            "cif_path", "allele", "peptide", "immunogenicity", "stem", "source_row_idx",
            "wt_peptide", "wt_cif_path",
        ],
    )
    print(
        f"Matched {len(df)} / {len(csv)} rows to PDBs "
        f"({dropped_mut} dropped: no mut PDB; {dropped_wt} dropped: no wt PDB)"
    )

    if args.strategy == "random":
        out_dir = os.path.join(SPLIT_BASE, f"splits_seed{args.seed}")
        train_df, val_df, test_df = split_random(df, args.seed)
    else:
        out_dir = os.path.join(SPLIT_BASE, f"splits_peptide_group_seed{args.seed}")
        train_df, val_df, test_df = split_peptide_group(df, args.seed)

    print(f"Strategy: {args.strategy}  seed: {args.seed}  out_dir: {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    splits = {"train": train_df, "val": val_df, "test": test_df}
    for name, split_df in splits.items():
        out_path = os.path.join(out_dir, f"{name}.csv")
        split_df.to_csv(out_path, index=False)
        unique_pep = split_df["peptide"].nunique()
        n_pos = int((split_df["immunogenicity"] == 1).sum())
        n_neg = int((split_df["immunogenicity"] == 0).sum())
        print(
            f"{name:5s}: {len(split_df):>6} rows  "
            f"[unique peptides: {unique_pep:>6}, pos: {n_pos}, neg: {n_neg}]"
        )
        print(f"          -> {out_path}")


if __name__ == "__main__":
    main()
