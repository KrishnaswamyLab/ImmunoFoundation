# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

**ImmunoStruct** (`immunofoundation` package) is a multimodal deep learning framework for immunogenicity prediction — predicting whether peptide-MHC (pMHC) complexes trigger a T-cell immune response. It jointly models three data modalities:
- **Sequence**: amino acid sequences via ESM-2 (frozen by default)
- **Structure**: 3D Cα coordinates parsed from CIF/PDB files
- **Biochemical**: 93 physicochemical descriptors (BLOSUM, Kidera factors, SASA, foreignness, etc.)

Training uses masked autoencoding (MAE) on all three modalities plus masked language modeling (MLM) on amino acid identity.

Associated paper: [bioRxiv 2024.11.01.621580](https://www.biorxiv.org/content/10.1101/2024.11.01.621580)

## Environment Setup

The project uses **uv** with Python 3.12 (pinned via `.python-version`).

```sh
uv venv -p 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
uv sync
```

On HPC with modules: `ml uv` before activating. If you get GLIBCXX errors, prepend a conda env's `lib/` to `LD_LIBRARY_PATH`.

`requirements.txt` pins `torch==2.4.0+cu124` and matching PyG wheels — CUDA version must match your environment. ESM-2 weights are downloaded automatically on first use.

## Training

Training is Hydra-configured. The main entrypoint is `train.py`.

```sh
# Monomer pretraining on AlphaFold DB (default config: configs/train_afdb.yaml)
python train.py

# Multimer / cancer pMHC fine-tuning
python train.py --config-name train_cancer

# Override any config key at the CLI
python train.py experiment.wandb.name=my_run experiment.num_devices=1

# SLURM (gpu_h200, 2 GPUs, 48h)
sbatch scripts/train_afdb.sh
```

**Important**: the `data.csv_path` fields in the configs point to `/home/hm638/...` (another user's home) and must be updated to local paths before training. The CSV must have a `cif_path` column pointing to CIF structure files.

For distributed training set these env vars (already in the SLURM script):
```sh
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29501
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
```

Wandb is used for experiment tracking. Set `experiment.wandb.name: null` to disable.

## Running Tests

Tests are standalone scripts (no pytest fixtures or conftest.py); run them directly:

```sh
python tests/test_environment.py       # import smoke test + version checks
python tests/test_fullmodel_monomer.py # monomer encode() forward pass
python tests/test_fullmodel_multimer.py
python tests/test_loss_mono.py         # monomer training_step()
python tests/test_loss_multi.py
```

**Known broken tests**: `test_biochem.py`, `test_sequence.py`, `test_structure.py`, and `test_dataloader.py` import legacy symbols (`ImmunoDataset`, `ESMSequenceModel`, `custom_collate`) that no longer exist in the codebase — they will fail with `ImportError`.

## Architecture

### Two Training Modes (controlled by `data.mono` in config)

**Monomer mode** (`ImmunoFoundationMonomerModule`, `train_afdb.yaml`):
- Single-chain CIF input → ESM-2 frozen encoder → `SequenceModel` (Transformer, 1280→32 dim) + `StructureModel` (Transformer, 3→32 dim)
- Losses: `recon_loss_seq` + `recon_loss_struct` + `mlm_loss`
- Batch keys: `coords [B,L,3]`, `adjs [B,L,L]`, `sequence [list]`, `masks [B,L]`

**Multimer mode** (`ImmunoFoundationMultimerModule`, `train_cancer.yaml`):
- Two-chain CIF input (chain A = peptide, chain B = MHC) + 93 biochemical scalars
- Same ESM/Sequence/Structure encoders applied to both chains + `BiochemicalModel` (MLP, 93→32)
- 7 loss terms covering seq/struct MAE for both chains + biochem recon + MLM×2
- Batch keys: prefixed `peptide_*` and `mhc_*`, plus `biochemical_properties [B,93]`

### Key Classes

| Class | File | Role |
|---|---|---|
| `ESM` | `models/components/ESM.py` | Wraps `esm2_t33_650M_UR50D`; extracts layer-33 reps (1280-dim) |
| `SequenceModel` | `models/components/SequenceModel.py` | TransformerEncoder on ESM reps → `out_dim` (default 32) |
| `StructureModel` | `models/components/StructureModel.py` | Projects 3D coords, TransformerEncoder with optional adj-masked attention |
| `BiochemicalModel` | `models/components/BiochemicalModel.py` | MLP on 93 biochemical scalars → `out_dim` |
| `ImmunoFoundationMonomerModule` | `models/ImmunoFoundationMonomerModule.py` | Lightning module for monomer MAE+MLM pretraining |
| `ImmunoFoundationMultimerModule` | `models/ImmunoFoundationMultimerModule.py` | Lightning module for pMHC multimer training |
| `ImmunoDataModule` | `data/ImmunoDataModule.py` | Routes to mono/multi datasets based on `data.mono` |
| `ImmunoMonomerDataset` | `data/components/ImmunoMonomerDataset.py` | Reads CSV, parses CIF, computes adjacency+masking on-the-fly |
| `ImmunoMultimerDataset` | `data/components/ImmunoMultimerDataset.py` | Same but for two-chain pMHC structures + biochemical features |

### Masking Strategy

Residues are masking candidates if they have fewer than `max_neighbors` contacts within `max_distance` Å. A `mask_rate` fraction of candidates are zeroed. Parameters live in the dataset/datamodule config.

### Data Not Included

CIF structure files, metadata CSVs, and the legacy `data/` immunogenicity datasets are gitignored and not in the repo. The shipped `data/` directory contains only HLA reference sequences and the IEDB immunogenicity label files.

## Configuration

Configs live in `configs/` and are loaded via Hydra. Key fields to update:
- `data.csv_path` — path to CSV with `cif_path` column (currently hardcoded to another user's path)
- `model.bio_chem.n_bio_prop: 93` — do not change; hardcoded to match `preprocess.py`
- Checkpoints saved to `ckpt/immunofoundation/<run_name>_<timestamp>/`
- Val metric monitored: `val/total_loss` (monomer), `train/loss` (cancer)
