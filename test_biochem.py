import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from omegaconf import OmegaConf

from immunofoundation.data.components.ImmunoDataset import ImmunoDataset
from immunofoundation.data.components.ImmunoDataset import custom_collate
from immunofoundation.models.components.BiochemicalModel import BiochemicalModel



def main():
    cfg = OmegaConf.load("configs/train.yaml")
    data_cfg = cfg.data
    bio_config = cfg.model.bio_chem
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
    bio_model = BiochemicalModel(bio_config)
    print("Initialized BioChemical model!!")
    bio_reps = bio_model(batch['biochemical_properties'])
    print(f"BioChemical Porperty embeddings: {bio_reps.shape}")
if __name__ == "__main__":
    main()