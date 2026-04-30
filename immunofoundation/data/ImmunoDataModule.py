
import torch
from typing import Any, Dict, Optional
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from immunofoundation.data.components.ImmunoMonomerDataset import ImmunoMonomerDataset, custom_collate_mono, custom_collate_mono_sparse
from immunofoundation.data.components.ImmunoMultimerDataset import ImmunoMultimerDataset, custom_collate_multi

class ImmunoDataModule(LightningDataModule):
    def __init__(self,data_cfg):
        super().__init__()
        self.data_cfg = data_cfg

        self.data_train = None
        self.data_val = None

        if self.data_cfg.mono:
            if getattr(self.data_cfg, 'sparse_batching', False):
                self.collate_fn = custom_collate_mono_sparse
            else:
                self.collate_fn = custom_collate_mono
        else:
            self.collate_fn = custom_collate_multi

    def setup(self, stage):
        if self.data_cfg.mono:
            self.data_train = ImmunoMonomerDataset(
                                self.data_cfg,
                                is_training=True,
                            )
            
            self.data_val = ImmunoMonomerDataset(
                                self.data_cfg,
                                is_training=False,
                            )
        else:
            self.data_train = ImmunoMultimerDataset(
                                self.data_cfg,
                                is_training=True,
                            )
            
            self.data_val = ImmunoMultimerDataset(
                                self.data_cfg,
                                is_training=False,
                            )
    def train_dataloader(self):
        num_workers = self.data_cfg.num_workers

        return DataLoader(
            self.data_train,
            batch_size = self.data_cfg.batch_size,
            num_workers = num_workers,
            prefetch_factor=None if num_workers == 0 else self.data_cfg.prefetch_factor,
            pin_memory=False,
            persistent_workers=True if num_workers > 0 else False,
            collate_fn=self.collate_fn
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size = self.data_cfg.batch_size,
            num_workers=2,
            prefetch_factor=2,
            persistent_workers=True,
            collate_fn=self.collate_fn
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
