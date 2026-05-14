
import torch
from typing import Any, Dict, Optional
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from immunofoundation.data.components.ImmunoMonomerDataset import ImmunoMonomerDataset, custom_collate_mono, custom_collate_mono_sparse
from immunofoundation.data.components.ImmunoMultimerDataset import ImmunoMultimerDataset, custom_collate_multi
from immunofoundation.data.components.ImmunoFinetuneDataset import ImmunoFinetuneDataset, custom_collate_finetune, custom_collate_finetune_cached

class ImmunoDataModule(LightningDataModule):
    def __init__(self,data_cfg):
        super().__init__()
        self.data_cfg = data_cfg

        self.data_train = None
        self.data_val = None
        self.data_test = None
        self.train_pos_weight = None
        self.train_data_len = None  # max source_row_idx + 1 over the train split (for libauc APLoss)

        self.is_finetune = bool(getattr(self.data_cfg, 'finetune', False))

        if self.is_finetune:
            if bool(getattr(self.data_cfg, "use_cached_embeddings", False)):
                self.collate_fn = custom_collate_finetune_cached
            else:
                self.collate_fn = custom_collate_finetune
        elif self.data_cfg.mono:
            if getattr(self.data_cfg, 'sparse_batching', False):
                self.collate_fn = custom_collate_mono_sparse
            else:
                self.collate_fn = custom_collate_mono
        else:
            self.collate_fn = custom_collate_multi

    def setup(self, stage):
        if self.is_finetune:
            self.data_train = ImmunoFinetuneDataset(self.data_cfg, split="train")
            self.data_val = ImmunoFinetuneDataset(self.data_cfg, split="val")
            self.data_test = ImmunoFinetuneDataset(self.data_cfg, split="test")
            self.train_pos_weight = self.data_train.pos_weight
            self.train_data_len = int(self.data_train.csv["source_row_idx"].max()) + 1
        elif self.data_cfg.mono:
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
            shuffle=True if self.is_finetune else False,
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

    def test_dataloader(self):
        if self.data_test is None:
            return None
        return DataLoader(
            self.data_test,
            batch_size=self.data_cfg.batch_size,
            num_workers=2,
            prefetch_factor=2,
            persistent_workers=True,
            shuffle=False,
            collate_fn=self.collate_fn,
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
