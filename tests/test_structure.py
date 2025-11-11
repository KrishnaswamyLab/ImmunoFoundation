import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from omegaconf import OmegaConf

from immunofoundation.data.components.ImmunoDataset import ImmunoDataset
from immunofoundation.data.components.ImmunoDataset import custom_collate
from immunofoundation.models.components.StructureModel import StructureModel



def main():
    cfg = OmegaConf.load("configs/train.yaml")
    data_cfg = cfg.data
    struct_config = cfg.model.structure
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = ImmunoDataset(
        data_cfg,
        is_training=True,
    )
    train_loader =  DataLoader(
            train_data,
            batch_size=data_cfg.batch_size,
            num_workers = data_cfg.num_workers,
            prefetch_factor=None if data_cfg.num_workers == 0 else data_cfg.prefetch_factor,
            pin_memory=False,
            persistent_workers=True if data_cfg.num_workers > 0 else False,
            collate_fn=custom_collate
        )
    batch = next(iter(train_loader))
    esm_model = StructureModel(struct_config)
    print("Initialized Structure model!!")
    peptide_reps, mhc_reps = esm_model(batch['peptide_coords'], batch['mhc_coords'])
    print(f"Peptide embeddings: {peptide_reps.shape}")
    print(f"MHC embeddings: {mhc_reps.shape}")
if __name__ == "__main__":
    main()