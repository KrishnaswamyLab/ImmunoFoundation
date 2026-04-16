#!/usr/bin/env python3
"""Build merged CSVs for finetuning from a PDB/mmCIF directory and a labels CSV.

Produces a merged CSV and optional train/val(/test) splits written to an output directory
so downstream training scripts can consume them directly.

Usage examples:
  scripts/build_finetune_csvs.py --pdb-dir /path/to/pdbs --labels-csv labels.csv --out-dir data/finetune
  scripts/build_finetune_csvs.py --pdb-dir /path/to/pdbs --out-dir data/finetune
"""
import argparse
import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split


def stem(path):
    base = os.path.basename(path)
    # remove common extensions
    for ext in ('.cif.gz', '.pdb.gz', '.cif', '.pdb'):
        if base.endswith(ext):
            return base[: -len(ext)]
    return os.path.splitext(base)[0]


def build_paths_df(pdb_dir):
    exts = ('**/*.cif', '**/*.cif.gz', '**/*.pdb', '**/*.pdb.gz')
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(pdb_dir, e), recursive=True))
    files = sorted(files)
    if len(files) == 0:
        raise ValueError(f'No structure files found under {pdb_dir}')
    df = pd.DataFrame({'cif_path': files})
    df['stem'] = df['cif_path'].apply(stem)
    return df


def try_merge(paths_df, labels_df, merge_on=None):
    # If user provided explicit merge key, use it
    if merge_on is not None:
        if merge_on not in labels_df.columns:
            raise ValueError(f"merge_on column '{merge_on}' not found in labels CSV")
        # try direct merge
        merged = paths_df.merge(labels_df, left_on='stem', right_on=merge_on, how='left')
        return merged

    # If labels contain `allele` and `mut_pep`, prefer constructing the filename stem as
    # "{allele}_{mut_pep}" which matches the PDB filenames in your dataset (e.g.
    # HLA-A*02:01_AAAAQQIQV.pdb).
    if 'allele' in labels_df.columns and 'mut_pep' in labels_df.columns:
        labels_df['_allele_mut_stem'] = labels_df['allele'].astype(str).str.strip() + '_' + labels_df['mut_pep'].astype(str).str.strip()
        merged = paths_df.merge(labels_df, left_on='stem', right_on='_allele_mut_stem', how='left')
        return merged

    # try to detect a column in labels that overlaps with stems
    best_col = None
    best_overlap = 0
    stems = set(paths_df['stem'].astype(str).unique())
    for c in labels_df.columns:
        vals = labels_df[c].astype(str).str.split('.').str[0].str.strip().unique()
        overlap = len(set(vals).intersection(stems))
        if overlap > best_overlap:
            best_overlap = overlap
            best_col = c

    if best_col is not None and best_overlap > 0:
        labels_df['_stem_candidate'] = labels_df[best_col].astype(str).str.split('.').str[0].str.strip()
        merged = paths_df.merge(labels_df, left_on='stem', right_on='_stem_candidate', how='left')
        return merged

    # fallback: if lengths match, concat by index
    if len(labels_df) == len(paths_df):
        merged = pd.concat([paths_df.reset_index(drop=True), labels_df.reset_index(drop=True)], axis=1)
        return merged

    raise ValueError('Could not automatically merge labels CSV to paths. Provide --merge-on to specify the key.')


def split_and_write(merged_df, out_dir, label_col='immunogenicity', train_size=0.8, test_size=0.1, random_state=42):
    os.makedirs(out_dir, exist_ok=True)
    merged_path = os.path.join(out_dir, 'merged.csv')
    merged_df.to_csv(merged_path, index=False)
    print('Wrote merged CSV ->', merged_path)

    if label_col in merged_df.columns and merged_df[label_col].notnull().any():
        labels = merged_df[label_col].astype(int)
        # compute splits
        if test_size > 0:
            train_val_idx, test_idx = train_test_split(merged_df.index.tolist(), test_size=test_size, stratify=labels, random_state=random_state)
            rel_val = (1.0 - train_size) / (1.0 - test_size)
            train_idx, val_idx = train_test_split(train_val_idx, test_size=rel_val, stratify=labels.iloc[train_val_idx], random_state=random_state)
        else:
            train_idx, val_idx = train_test_split(merged_df.index.tolist(), test_size=(1.0 - train_size), stratify=labels, random_state=random_state)
            test_idx = []

        train_df = merged_df.loc[train_idx].reset_index(drop=True)
        val_df = merged_df.loc[val_idx].reset_index(drop=True)
        train_df.to_csv(os.path.join(out_dir, 'train.csv'), index=False)
        val_df.to_csv(os.path.join(out_dir, 'val.csv'), index=False)
        print(f'Wrote train ({len(train_df)} rows) and val ({len(val_df)} rows) CSVs to', out_dir)
        if len(test_idx) > 0:
            test_df = merged_df.loc[test_idx].reset_index(drop=True)
            test_df.to_csv(os.path.join(out_dir, 'test.csv'), index=False)
            print('Wrote test CSV ->', os.path.join(out_dir, 'test.csv'))
    else:
        print(f"Label column '{label_col}' not present or empty in merged CSV; only wrote merged paths.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb-dir', required=True)
    parser.add_argument('--labels-csv', default=None)
    parser.add_argument('--merge-on', default=None, help='Column in labels CSV to match to filename stem')
    parser.add_argument('--out-dir', default='data/finetune', help='Directory to write merged and split CSVs')
    parser.add_argument('--label-col', default='immunogenicity')
    parser.add_argument('--train-size', type=float, default=0.8)
    parser.add_argument('--test-size', type=float, default=0.1)
    args = parser.parse_args()

    paths_df = build_paths_df(args.pdb_dir)
    print(f'Found {len(paths_df)} structure files under {args.pdb_dir}')

    if args.labels_csv is None:
        # just write paths CSV
        os.makedirs(args.out_dir, exist_ok=True)
        p = os.path.join(args.out_dir, 'paths.csv')
        paths_df.to_csv(p, index=False)
        print('Wrote paths CSV ->', p)
        return

    labels_df = pd.read_csv(args.labels_csv)
    merged = try_merge(paths_df, labels_df, merge_on=args.merge_on)
    split_and_write(merged, args.out_dir, label_col=args.label_col, train_size=args.train_size, test_size=args.test_size)


if __name__ == '__main__':
    main()
