"""
Code adapted from
https://github.com/rish-16/rna-backbone-design/blob/main/train_se3_flows.py
"""

import os
import GPUtil
import torch
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from datetime import datetime

from pytorch_lightning import Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

# IMPORT PROJECT MODULES
from immunofoundation.models.ImmunoFoundationMonomerModule import ImmunoFoundationMonomerModule
from immunofoundation.models.ImmunoFoundationMultimerModule import ImmunoFoundationMultimerModule
from immunofoundation.data.ImmunoDataModule import ImmunoDataModule

import immunofoundation.utils as eu 


log = eu.get_pylogger(__name__)
torch.set_float32_matmul_precision('high')

class Experiment:
    def __init__(self, *, cfg: DictConfig):
        self._cfg = cfg
        self._data_cfg = cfg.data
        self._exp_cfg = cfg.experiment
        if self._data_cfg.mono:
            self._model = ImmunoFoundationMonomerModule(self._cfg.model)
        else:
            self._model = ImmunoFoundationMultimerModule(self._cfg.model)
        self._datamodule = ImmunoDataModule(self._cfg.data)
 
    def train(self):
        callbacks = []
        
        # Initialize wandb logger if name is specified
        if self._exp_cfg.wandb.get("name"):
            logger = WandbLogger(**self._exp_cfg.wandb)
            log.info(f"Wandb initialized: {self._exp_cfg.wandb.name}")
        else:
            logger = None
            log.info("Wandb disabled (no name specified)")

        if not self._exp_cfg.debug:
            # Checkpoint directory
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            ckpt_dir = self._exp_cfg.checkpointer.dirpath+"_"+run_id
            os.makedirs(ckpt_dir, exist_ok=True)
            log.info(f"Checkpoints saved to {ckpt_dir}")

            # Model checkpoints
            callbacks.append(ModelCheckpoint(**self._exp_cfg.checkpointer))

            # Save config
            cfg_path = os.path.join(ckpt_dir, 'config.yaml')
            with open(cfg_path, 'w') as f:
                OmegaConf.save(config=self._cfg, f=f.name)

            # Log config to wandb
            if logger is not None:
                cfg_dict = OmegaConf.to_container(self._cfg, resolve=True)
                flat_cfg = dict(eu.flatten_dict(cfg_dict))
                if isinstance(logger.experiment.config, wandb.sdk.wandb_config.Config):
                    logger.experiment.config.update(flat_cfg)

        num_gpus = torch.cuda.device_count()
        devices = min(num_gpus, self._exp_cfg.num_devices)
        log.info(f"Using {devices} device(s) out of {num_gpus} visible")
        
        trainer = Trainer(
            **self._exp_cfg.trainer,
            callbacks=callbacks,
            logger=logger,
            use_distributed_sampler=True,
            enable_progress_bar=True,
            enable_model_summary=True,
            devices=devices,
        )

        trainer.fit(
            model=self._model,
            datamodule=self._datamodule,
        )

@hydra.main(version_base=None, config_path="./configs", config_name="train_afdb")
def main(cfg: DictConfig):

    exp = Experiment(cfg=cfg)
    exp.train()

if __name__ == "__main__":
    main()