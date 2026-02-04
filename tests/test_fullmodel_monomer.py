import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from omegaconf import OmegaConf

from immunofoundation.data.components.ImmunoMonomerDataset import ImmunoMonomerDataset
from immunofoundation.data.components.ImmunoMonomerDataset import custom_collate
from immunofoundation.models.ImmunoFoundationMonomerModule import ImmunoFoundationMonomerModule



def main():
    cfg = OmegaConf.load("configs/train_afdb.yaml")
    data_cfg = cfg.data
    model_cfg = cfg.model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = ImmunoMonomerDataset(
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
    model = ImmunoFoundationMonomerModule(model_cfg)
    print("Initialized ImmunoFoundationMonomerModule model!!")
    seq_embeddings, struct_embeddings = model.encode(batch)
    print(f"seq_embeddings: {seq_embeddings.shape}")
    print(f"struct_embeddings: {struct_embeddings.shape}")
if __name__ == "__main__":
    main()