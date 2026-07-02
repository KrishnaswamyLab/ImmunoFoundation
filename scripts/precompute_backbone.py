"""Precompute frozen-backbone embeddings for fast fine-tune iteration.

For each unique CIF across the train/val/test split CSVs, run the pretrained backbone
(aa_embedding_model + sequence_model + structure_model) once and cache the per-residue
embeddings. With these cached, fine-tuning with `model.freeze_backbone=true` only needs to run
the cross-modal fusion + head — typically 50-100x faster per step.

Output: a single torch dict at .cache/iedb_finetune/embeddings_<ckpt_hash>.pt
"""

import argparse
import hashlib
import os
import pickle
import sys

import pandas as pd
import torch
from omegaconf import OmegaConf
from sklearn.neighbors import kneighbors_graph
from tqdm import tqdm

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

from immunofoundation.data.components.preprocess_pdb import extract_ca_and_sequence_pmhc, normalize_coords
from immunofoundation.models.components.ESM import ESM
from immunofoundation.models.components.SequenceModel import SequenceModel
from immunofoundation.models.components.StructureModel import StructureModel


def cache_path_for(ckpt_path, cache_subdir="iedb_finetune"):
    cache_dir = os.path.join(REPO, ".cache", cache_subdir)
    os.makedirs(cache_dir, exist_ok=True)
    h = hashlib.md5(os.path.abspath(ckpt_path).encode()).hexdigest()[:16]
    return os.path.join(cache_dir, f"embeddings_{h}.pt")


def collect_unique_cifs(split_csv_paths):
    cifs = set()
    include_wt = False
    for split, path in split_csv_paths.items():
        df = pd.read_csv(path)
        cifs.update(df["cif_path"].tolist())
        if "wt_cif_path" in df.columns:
            include_wt = True
            cifs.update(df["wt_cif_path"].tolist())
            print(
                f"  {split}: {len(df)} rows, "
                f"{df['cif_path'].nunique()} unique mut CIFs + {df['wt_cif_path'].nunique()} unique wt CIFs"
            )
        else:
            print(f"  {split}: {len(df)} rows, {df['cif_path'].nunique()} unique CIFs")
    print(f"  union: {len(cifs)} unique CIFs ({'incl. wildtype' if include_wt else 'mut only'})")
    return sorted(cifs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_finetune_iedb.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--k", type=int, default=15, help="kNN neighbors for adjacency")
    parser.add_argument("--init-checkpoint", default=None, help="Override cfg.init_checkpoint")
    args = parser.parse_args()

    cfg = OmegaConf.load(os.path.join(REPO, args.config))
    if args.init_checkpoint:
        OmegaConf.set_struct(cfg, False)
        cfg.init_checkpoint = args.init_checkpoint
    cache_subdir = getattr(cfg.data, "cache_subdir", None) or "iedb_finetune"
    out_path = cache_path_for(cfg.init_checkpoint, cache_subdir=cache_subdir)
    cache = {}
    if os.path.exists(out_path):
        print(f"[precompute] loading existing cache: {out_path}")
        cache = torch.load(out_path, map_location="cpu", weights_only=False)
        print(f"             {len(cache)} entries already cached")

    pep_chain = getattr(cfg.data.structure, "peptide_chain", "P")
    mhc_chain = getattr(cfg.data.structure, "mhc_chain", "A")
    print(f"[precompute] using chains peptide={pep_chain}, mhc={mhc_chain}")

    print("[precompute] collecting unique CIFs across splits")
    cifs = collect_unique_cifs(dict(cfg.data.split_csv_paths))
    cifs = [c for c in cifs if c not in cache]
    if not cifs:
        print("[precompute] all CIFs already cached; nothing to do")
        return
    print(f"[precompute] {len(cifs)} CIFs to embed")

    print(f"[precompute] building backbone on {args.device}")
    aa = ESM(cfg.model.sequence).to(args.device).eval()
    seqmod = SequenceModel(cfg.model.sequence).to(args.device).eval()
    strmod = StructureModel(cfg.model.structure).to(args.device).eval()

    print(f"[precompute] loading weights from {cfg.init_checkpoint}")
    sd = torch.load(cfg.init_checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    aa_sd = {k.removeprefix("aa_embedding_model."): v for k, v in sd.items() if k.startswith("aa_embedding_model.")}
    seq_sd = {k.removeprefix("sequence_model."): v for k, v in sd.items() if k.startswith("sequence_model.")}
    str_sd = {k.removeprefix("structure_model."): v for k, v in sd.items() if k.startswith("structure_model.")}
    aa.load_state_dict(aa_sd, strict=False)
    seqmod.load_state_dict(seq_sd, strict=False)
    strmod.load_state_dict(str_sd, strict=False)

    parse_failures = 0
    with torch.no_grad():
        for cif in tqdm(cifs, desc="encoding"):
            try:
                pep_coords_raw, pep_seq, mhc_coords_raw, mhc_seq = extract_ca_and_sequence_pmhc(
                    cif, peptide_chain=pep_chain, mhc_chain=mhc_chain
                )
            except Exception:
                parse_failures += 1
                continue
            if not pep_seq or not mhc_seq:
                parse_failures += 1
                continue
            if len(pep_seq) != len(pep_coords_raw) or len(mhc_seq) != len(mhc_coords_raw):
                parse_failures += 1
                continue

            pep_esm = aa([pep_seq])
            mhc_esm = aa([mhc_seq])
            pep_z_seq = seqmod(pep_esm).squeeze(0).cpu()  # [Lp, 32]
            mhc_z_seq = seqmod(mhc_esm).squeeze(0).cpu()  # [Lm, 32]

            coords_pep = normalize_coords(pep_coords_raw)
            coords_mhc = normalize_coords(mhc_coords_raw)
            k_pep = max(1, min(args.k, len(pep_seq) - 1))
            adj_pep = kneighbors_graph(coords_pep, n_neighbors=k_pep).toarray()
            adj_mhc = kneighbors_graph(coords_mhc, n_neighbors=args.k).toarray()
            adj_pep_t = torch.tensor(adj_pep, dtype=torch.float32, device=args.device).unsqueeze(0)
            adj_mhc_t = torch.tensor(adj_mhc, dtype=torch.float32, device=args.device).unsqueeze(0)
            coords_pep_t = torch.tensor(coords_pep, dtype=torch.float32, device=args.device).unsqueeze(0)
            coords_mhc_t = torch.tensor(coords_mhc, dtype=torch.float32, device=args.device).unsqueeze(0)
            pep_z_str = strmod(adj_pep_t, coords_pep_t).squeeze(0).cpu()  # [Lp, 32]
            mhc_z_str = strmod(adj_mhc_t, coords_mhc_t).squeeze(0).cpu()  # [Lm, 32]

            cache[cif] = {
                "peptide_sequence": pep_seq,
                "mhc_sequence": mhc_seq,
                "peptide_len": len(pep_seq),
                "mhc_len": len(mhc_seq),
                "peptide_z_seq": pep_z_seq,
                "peptide_z_str": pep_z_str,
                "mhc_z_seq": mhc_z_seq,
                "mhc_z_str": mhc_z_str,
            }

    print(f"[precompute] encoded {len(cache)} CIFs ({parse_failures} parse failures)")
    print(f"[precompute] saving to {out_path}")
    torch.save(cache, out_path)
    print(f"[precompute] done. cache size: {os.path.getsize(out_path) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
