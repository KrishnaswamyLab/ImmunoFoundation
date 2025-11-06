import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from omegaconf import OmegaConf

from immunofoundation.data.components.ImmunoDataset import ImmunoDataset
from immunofoundation.data.components.ImmunoDataset import custom_collate
from immunofoundation.models.ImmunoFoundationModule import ImmunoFoundationModule



def main():
    cfg = OmegaConf.load("configs/train.yaml")
    data_cfg = cfg.data
    model_cfg = cfg.model
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
    model = ImmunoFoundationModule(model_cfg)
    print("Initialized ImmunoFoundationModel model!!")
    peptide_seq_embeddings, mhc_seq_embeddings, peptide_struct_embeddings, mhc_struct_embeddings, bio_chem_embeddings = model(batch)
    print(f"peptide_seq_embeddings: {peptide_seq_embeddings.shape}")
    print(f"mhc_seq_embeddings: {mhc_seq_embeddings.shape}")
    print(f"peptide_struct_embeddings: {peptide_struct_embeddings.shape}")
    print(f"mhc_struct_embeddings: {mhc_struct_embeddings.shape}")
    print(f"bio_chem_embeddings: {bio_chem_embeddings.shape}")
if __name__ == "__main__":
    main()