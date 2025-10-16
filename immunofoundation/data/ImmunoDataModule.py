
import torch
from typing import Any, Dict, Optional
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from immunofoundation.data.components.ImmunoDataset import ImmunoDataset


class ImmunoDataModule(LightningDataModule):
    def __init__(self,data_cfg):
        super().__init__()
        self.data_cfg = data_cfg

        self.data_train = None
        self.data_val = None

    def setup(self, stage):
        self.data_train = ImmunoDataset(
                            self.data_cfg,
                            is_training=True,
                        )
        
        self.data_val = ImmunoDataset(
                            self.data_cfg,
                            is_training=False,
                        )

    def train_dataloader(self):
        num_workers = self.data_cfg.num_workers

        return DataLoader(
            self.data_train,
            num_workers = num_workers,
            prefetch_factor=None if num_workers == 0 else self.data_cfg.prefetch_factor,
            pin_memory=False,
            persistent_workers=True if num_workers > 0 else False,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            num_workers=2,
            prefetch_factor=2,
            persistent_workers=True
        )

    def prepare_data(self):
        """Download data if needed.
        Do not use it to assign state (self.x = y).
        """
        pass

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass