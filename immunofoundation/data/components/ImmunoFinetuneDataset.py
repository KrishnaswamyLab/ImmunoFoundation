import hashlib
import os
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import kneighbors_graph
from torch.utils.data import Dataset
from tqdm import tqdm

from immunofoundation.data.components.ImmunoMultimerDataset import (
    ImmunoMultimerDataset,
    custom_collate_multi,
    pad,
)
from immunofoundation.data.components.preprocess_pdb import (
    extract_ca_and_sequence_pmhc,
    normalize_coords,
)


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def _cache_dir(data_cfg):
    sub = getattr(data_cfg, "cache_subdir", None) or "iedb_finetune"
    base = os.path.join(_REPO_ROOT, ".cache", sub)
    os.makedirs(base, exist_ok=True)
    return base


class ImmunoFinetuneDataset(ImmunoMultimerDataset):
    """Fine-tune dataset reading from per-split CSVs built by scripts/build_iedb_splits.py.

    Splits come from data.split_csv_paths.{train,val,test}. Each split CSV has columns
    [cif_path, allele, peptide, immunogenicity, stem, source_row_idx]. AF3 outputs chain
    A=MHC, chain P=peptide (validated against data/IEDB/af3_folded_pdbs/), so we use
    extract_ca_and_sequence_pmhc rather than the parent's extract_ca_and_sequence.

    Adds `immunogenicity` to the per-sample dict.
    """

    SPLITS = ("train", "val", "test")
    _MEM_CACHE = {}  # path -> filtered DataFrame
    _EMB_CACHE = {}  # path -> dict[cif_path -> embedding dict]

    def __init__(self, data_cfg, split):
        if split not in self.SPLITS:
            raise ValueError(f"split must be one of {self.SPLITS}, got {split!r}")
        self.split = split
        self.data_cfg = data_cfg
        self.is_training = split == "train"
        self.use_cached_embeddings = bool(getattr(data_cfg, "use_cached_embeddings", False))
        self.contrastive = bool(getattr(data_cfg, "contrastive_wildtype", False))
        if self.use_cached_embeddings:
            self.embeddings_cache = self._load_embeddings_cache()
        else:
            self.embeddings_cache = None
        self._init_metadata()

    def _embeddings_cache_path(self, ckpt_path):
        h = hashlib.md5(os.path.abspath(ckpt_path).encode()).hexdigest()[:16]
        return os.path.join(_cache_dir(self.data_cfg), f"embeddings_{h}.pt")

    def _load_embeddings_cache(self):
        ckpt = self.data_cfg.get("init_checkpoint_for_cache", None) or os.environ.get("IF_CACHE_CKPT")
        if ckpt is None:
            raise ValueError(
                "use_cached_embeddings=true but no init_checkpoint provided. "
                "Either set data.init_checkpoint_for_cache or pass IF_CACHE_CKPT env var "
                "matching the checkpoint used to build the cache (typically the top-level cfg.init_checkpoint)."
            )
        cache_path = self._embeddings_cache_path(ckpt)
        if cache_path in ImmunoFinetuneDataset._EMB_CACHE:
            return ImmunoFinetuneDataset._EMB_CACHE[cache_path]
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Backbone embeddings cache not found: {cache_path}\n"
                f"Run: python scripts/precompute_backbone.py"
            )
        print(f"[ImmunoFinetuneDataset] loading embeddings cache: {cache_path}", flush=True)
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        ImmunoFinetuneDataset._EMB_CACHE[cache_path] = cache
        return cache

    def _filtered_cache_path(self, csv_path):
        mtime = int(os.path.getmtime(csv_path))
        salt = "contrastive=1" if self.contrastive else "contrastive=0"
        key = hashlib.md5(f"{os.path.abspath(csv_path)}|{mtime}|{salt}".encode()).hexdigest()[:16]
        return os.path.join(_cache_dir(self.data_cfg), f"filtered_{key}.pkl")

    def _validate_cif(self, cif, pep_chain, mhc_chain):
        try:
            pep_coords, pep_seq, mhc_coords, mhc_seq = extract_ca_and_sequence_pmhc(
                cif, peptide_chain=pep_chain, mhc_chain=mhc_chain
            )
        except Exception:
            return False
        if len(pep_seq) == 0 or len(mhc_seq) == 0:
            return False
        if len(pep_seq) != len(pep_coords) or len(mhc_seq) != len(mhc_coords):
            return False
        return True

    def _load_filtered(self, csv_path):
        """Read split CSV and drop rows whose CIF fails parse or seq/coord length mismatch.

        When contrastive=True, also requires `wt_cif_path` to validate.
        """
        if csv_path in ImmunoFinetuneDataset._MEM_CACHE:
            return ImmunoFinetuneDataset._MEM_CACHE[csv_path]

        cache_path = self._filtered_cache_path(csv_path)
        if os.path.exists(cache_path):
            print(f"[ImmunoFinetuneDataset] Loading filtered CSV from cache: {cache_path}", flush=True)
            with open(cache_path, "rb") as f:
                filtered = pickle.load(f)
        else:
            full = pd.read_csv(csv_path)
            if self.contrastive and "wt_cif_path" not in full.columns:
                raise ValueError(
                    f"contrastive_wildtype=true but split CSV {csv_path} has no wt_cif_path column. "
                    "Re-run scripts/build_cedar_cancer_splits.py."
                )
            pep_chain = getattr(self.data_cfg.structure, "peptide_chain", "P")
            mhc_chain = getattr(self.data_cfg.structure, "mhc_chain", "A")
            unique_cifs = full["cif_path"].drop_duplicates().tolist()
            cifs_to_check = list(unique_cifs)
            if self.contrastive:
                wt_unique = full["wt_cif_path"].drop_duplicates().tolist()
                cifs_to_check = list({*unique_cifs, *wt_unique})
            print(
                f"[ImmunoFinetuneDataset] Validating {len(cifs_to_check)} unique CIFs "
                f"({len(full)} rows{'; contrastive incl. wt' if self.contrastive else ''}) from {csv_path}",
                flush=True,
            )
            valid_cifs = set()
            for cif in tqdm(cifs_to_check, desc=f"Validating {os.path.basename(csv_path)}"):
                if self._validate_cif(cif, pep_chain, mhc_chain):
                    valid_cifs.add(cif)
            mask = full["cif_path"].isin(valid_cifs)
            if self.contrastive:
                mask &= full["wt_cif_path"].isin(valid_cifs)
            filtered = full[mask].reset_index(drop=True)
            dropped = len(full) - len(filtered)
            print(
                f"[ImmunoFinetuneDataset] Kept {len(filtered)} / {len(full)} rows "
                f"({len(valid_cifs)} / {len(cifs_to_check)} unique CIFs valid; {dropped} rows dropped)",
                flush=True,
            )
            with open(cache_path, "wb") as f:
                pickle.dump(filtered, f)
            print(f"[ImmunoFinetuneDataset] Cached to {cache_path}", flush=True)

        ImmunoFinetuneDataset._MEM_CACHE[csv_path] = filtered
        return filtered

    def _resolve_split_path(self, split):
        paths = self.data_cfg.split_csv_paths
        path = paths[split] if split in paths else getattr(paths, split)
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        return path

    def _init_metadata(self):
        split_path = self._resolve_split_path(self.split)
        filtered = self._load_filtered(split_path)
        self.csv = filtered.reset_index(drop=True)
        self.raw_csv = self.csv

        # pos_weight = neg / pos over the train split (same value across train/val/test instances).
        train_path = self._resolve_split_path("train")
        train_filtered = self._load_filtered(train_path) if train_path != split_path else filtered
        train_labels = train_filtered["immunogenicity"].astype(int).to_numpy()
        n_pos = int((train_labels == 1).sum())
        n_neg = int((train_labels == 0).sum())
        self.pos_weight = float(n_neg) / max(n_pos, 1)

        print(f"[ImmunoFinetuneDataset:{self.split}] {len(self.csv)} samples, train pos_weight={self.pos_weight:.3f}", flush=True)

    def _build_complex_features(self, cif_path):
        """Read one PDB and produce {peptide,mhc}_{len,sequence,coords,mask,adj}."""
        pep_chain = getattr(self.data_cfg.structure, "peptide_chain", "P")
        mhc_chain = getattr(self.data_cfg.structure, "mhc_chain", "A")
        pep_coords_raw, pep_seq, mhc_coords_raw, mhc_seq = extract_ca_and_sequence_pmhc(
            cif_path, peptide_chain=pep_chain, mhc_chain=mhc_chain
        )
        feats = {
            "peptide_len": len(pep_seq),
            "mhc_len": len(mhc_seq),
            "peptide_sequence": pep_seq,
            "mhc_sequence": mhc_seq,
        }
        raw_pep_tensor = torch.tensor(pep_coords_raw).float()
        raw_mhc_tensor = torch.tensor(mhc_coords_raw).float()
        peptide_distances = torch.cdist(raw_pep_tensor, raw_pep_tensor, 2)
        mhc_distances = torch.cdist(raw_mhc_tensor, raw_mhc_tensor, 2)
        feats["peptide_mask"] = self.mask_residues(
            (peptide_distances < self.data_cfg.mask.max_distance).sum(1) < self.data_cfg.mask.max_neighbors
        )
        feats["mhc_mask"] = self.mask_residues(
            (mhc_distances < self.data_cfg.mask.max_distance).sum(1) < self.data_cfg.mask.max_neighbors
        )
        coords_peptide = normalize_coords(pep_coords_raw)
        coords_mhc = normalize_coords(mhc_coords_raw)
        feats["peptide_coords"] = torch.tensor(coords_peptide).float()
        feats["mhc_coords"] = torch.tensor(coords_mhc).float()
        if self.data_cfg.structure.adj:
            k_pep = min(self.data_cfg.structure.k, len(pep_seq) - 1)
            feats["peptide_adj"] = kneighbors_graph(coords_peptide, n_neighbors=max(k_pep, 1))
            feats["mhc_adj"] = kneighbors_graph(coords_mhc, n_neighbors=self.data_cfg.structure.k)
        else:
            feats["peptide_adj"] = None
            feats["mhc_adj"] = None
        return feats

    def _process_csv_row(self, csv_row):
        if self.use_cached_embeddings:
            return self._process_cached(csv_row)
        final_features = self._build_complex_features(csv_row["cif_path"])
        if self.contrastive:
            wt_feats = self._build_complex_features(csv_row["wt_cif_path"])
            for k, v in wt_feats.items():
                final_features[f"wt_{k}"] = v
        final_features["immunogenicity"] = float(csv_row["immunogenicity"])
        final_features["index"] = int(csv_row["source_row_idx"])
        return final_features

    def _lookup_cached(self, cif):
        entry = self.embeddings_cache.get(cif)
        if entry is None:
            raise KeyError(
                f"cif_path missing from embeddings cache: {cif}\n"
                f"Re-run scripts/precompute_backbone.py to refresh."
            )
        return {
            "peptide_len": entry["peptide_len"],
            "mhc_len": entry["mhc_len"],
            "peptide_z_seq": entry["peptide_z_seq"],
            "peptide_z_str": entry["peptide_z_str"],
            "mhc_z_seq": entry["mhc_z_seq"],
            "mhc_z_str": entry["mhc_z_str"],
        }

    def _process_cached(self, csv_row):
        out = self._lookup_cached(csv_row["cif_path"])
        if self.contrastive:
            wt = self._lookup_cached(csv_row["wt_cif_path"])
            for k, v in wt.items():
                out[f"wt_{k}"] = v
        out["immunogenicity"] = float(csv_row["immunogenicity"])
        out["index"] = int(csv_row["source_row_idx"])
        return out


def _pad_emb(t, target_len):
    """Right-pad a [L, D] tensor with zeros to [target_len, D]."""
    L, D = t.shape
    if L == target_len:
        return t
    out = torch.zeros(target_len, D, dtype=t.dtype)
    out[:L] = t
    return out


def _collate_cached_side(batch_list, prefix=""):
    """Pad a side (mut or wt) of cached embeddings + build pad masks. Returns prefixed keys."""
    plen = f"{prefix}peptide_len"
    mlen = f"{prefix}mhc_len"
    max_lp = max(x[plen] for x in batch_list)
    max_lm = max(x[mlen] for x in batch_list)
    out = {
        f"{prefix}peptide_z_seq": torch.stack([_pad_emb(rec[f"{prefix}peptide_z_seq"], max_lp) for rec in batch_list]),
        f"{prefix}peptide_z_str": torch.stack([_pad_emb(rec[f"{prefix}peptide_z_str"], max_lp) for rec in batch_list]),
        f"{prefix}mhc_z_seq": torch.stack([_pad_emb(rec[f"{prefix}mhc_z_seq"], max_lm) for rec in batch_list]),
        f"{prefix}mhc_z_str": torch.stack([_pad_emb(rec[f"{prefix}mhc_z_str"], max_lm) for rec in batch_list]),
        f"{prefix}peptide_pad_mask": torch.stack([
            torch.cat([
                torch.zeros(rec[plen], dtype=torch.bool),
                torch.ones(max_lp - rec[plen], dtype=torch.bool),
            ])
            for rec in batch_list
        ]),
        f"{prefix}mhc_pad_mask": torch.stack([
            torch.cat([
                torch.zeros(rec[mlen], dtype=torch.bool),
                torch.ones(max_lm - rec[mlen], dtype=torch.bool),
            ])
            for rec in batch_list
        ]),
    }
    return out


def custom_collate_finetune_cached(batch_list):
    """Collate for the cached-embeddings fast path. No CIF tensors, just embeddings + masks + label."""
    out = _collate_cached_side(batch_list, prefix="")
    if "wt_peptide_len" in batch_list[0]:
        out.update(_collate_cached_side(batch_list, prefix="wt_"))
    out["immunogenicity"] = torch.tensor([rec["immunogenicity"] for rec in batch_list], dtype=torch.float)
    out["index"] = torch.tensor([rec["index"] for rec in batch_list], dtype=torch.long)
    return out


_MUT_FEATURE_KEYS = (
    "peptide_len", "mhc_len",
    "peptide_sequence", "mhc_sequence",
    "peptide_coords", "mhc_coords",
    "peptide_mask", "mhc_mask",
    "peptide_adj", "mhc_adj",
)


def _build_pad_masks(batch_list):
    max_len_peptide = max(x["peptide_len"] for x in batch_list)
    max_len_mhc = max(x["mhc_len"] for x in batch_list)
    peptide_pad_mask = torch.stack([
        torch.cat([
            torch.zeros(rec["peptide_len"], dtype=torch.bool),
            torch.ones(max_len_peptide - rec["peptide_len"], dtype=torch.bool),
        ])
        for rec in batch_list
    ])
    mhc_pad_mask = torch.stack([
        torch.cat([
            torch.zeros(rec["mhc_len"], dtype=torch.bool),
            torch.ones(max_len_mhc - rec["mhc_len"], dtype=torch.bool),
        ])
        for rec in batch_list
    ])
    return peptide_pad_mask, mhc_pad_mask


def custom_collate_finetune(batch_list):
    base = custom_collate_multi(batch_list)
    peptide_pad_mask, mhc_pad_mask = _build_pad_masks(batch_list)
    base["peptide_pad_mask"] = peptide_pad_mask
    base["mhc_pad_mask"] = mhc_pad_mask
    base["immunogenicity"] = torch.tensor(
        [rec["immunogenicity"] for rec in batch_list], dtype=torch.float
    )
    base["index"] = torch.tensor([rec["index"] for rec in batch_list], dtype=torch.long)

    if "wt_peptide_len" in batch_list[0]:
        wt_records = [
            {k: rec[f"wt_{k}"] for k in _MUT_FEATURE_KEYS}
            for rec in batch_list
        ]
        wt_base = custom_collate_multi(wt_records)
        wt_peptide_pad_mask, wt_mhc_pad_mask = _build_pad_masks(wt_records)
        for k, v in wt_base.items():
            base[f"wt_{k}"] = v
        base["wt_peptide_pad_mask"] = wt_peptide_pad_mask
        base["wt_mhc_pad_mask"] = wt_mhc_pad_mask

    return base
