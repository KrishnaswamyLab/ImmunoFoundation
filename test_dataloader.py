from immunofoundation.data.components.ImmunoDataset import ImmunoDataset
from immunofoundation.data.components.ImmunoDataset import custom_collate
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from types import SimpleNamespace
from omegaconf import OmegaConf


def main():
    cfg = OmegaConf.load("configs/train.yaml")
    data_cfg = cfg.data
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
if __name__ == "__main__":
    main()