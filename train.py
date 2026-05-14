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

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

# IMPORT PROJECT MODULES
from immunofoundation.models.ImmunoFoundationMonomerModule import ImmunoFoundationMonomerModule
from immunofoundation.models.ImmunoFoundationMultimerModule import ImmunoFoundationMultimerModule
from immunofoundation.models.ImmunoFoundationFinetuneModule import ImmunoFoundationFinetuneModule
from immunofoundation.data.ImmunoDataModule import ImmunoDataModule

import immunofoundation.utils as eu 


log = eu.get_pylogger(__name__)
torch.set_float32_matmul_precision('high')

class Experiment:
    def __init__(self, *, cfg: DictConfig):
        self._cfg = cfg
        self._data_cfg = cfg.data
        self._exp_cfg = cfg.experiment
        seed_everything(int(self._data_cfg.get("seed", 0)))
        self._is_finetune = bool(self._data_cfg.get("finetune", False))

        # If using cached backbone embeddings, surface init_checkpoint to the dataset so it can
        # locate the cache file built by scripts/precompute_backbone.py.
        if self._is_finetune and bool(self._data_cfg.get("use_cached_embeddings", False)):
            OmegaConf.set_struct(self._cfg.data, False)
            self._cfg.data.init_checkpoint_for_cache = self._cfg.init_checkpoint
            OmegaConf.set_struct(self._cfg.data, True)

        self._datamodule = ImmunoDataModule(self._cfg.data)

        if self._is_finetune:
            self._datamodule.setup("fit")
            OmegaConf.set_struct(self._cfg.model, False)
            if getattr(self._cfg.model, "pos_weight", None) is None:
                self._cfg.model.pos_weight = float(self._datamodule.train_pos_weight)
            self._cfg.model.train_data_len = int(self._datamodule.train_data_len)
            OmegaConf.set_struct(self._cfg.model, True)
            self._model = ImmunoFoundationFinetuneModule(self._cfg.model)
        elif self._data_cfg.mono:
            self._model = ImmunoFoundationMonomerModule(self._cfg.model)
        else:
            self._model = ImmunoFoundationMultimerModule(self._cfg.model)

        init_ckpt = cfg.get("init_checkpoint", None)
        if init_ckpt:
            ckpt = torch.load(init_ckpt, map_location="cpu")
            # strict=False is required for cross-module transfer (e.g. monomer → multimer →
            # finetune) because the target module has keys absent from the source checkpoint
            # (fusion, head for finetune; bio_model, biochem_decoder for multimer).
            missing, unexpected = self._model.load_state_dict(ckpt["state_dict"], strict=False)
            log.info(f"Loaded weights from checkpoint: {init_ckpt}")
            if self._is_finetune:
                expected_prefixes = ("fusion.", "head.", "pos_weight_buf", "loss.pos_weight", "eval_loss.pos_weight", "train_loss.")
                expected_missing = [k for k in missing if k.startswith(expected_prefixes)]
                unexpected_missing = [k for k in missing if not k.startswith(expected_prefixes)]
                log.info("=" * 60)
                log.info(f"[Fine-tune] Pretrained weights loaded from {init_ckpt}")
                log.info(f"[Fine-tune] Missing (random init, expected): {len(expected_missing)} keys under fusion./head.")
                log.info(f"[Fine-tune] Missing (UNEXPECTED): {unexpected_missing}")
                log.info(f"[Fine-tune] Unexpected (ignored, pretraining decoders): {unexpected}")
                log.info("=" * 60)
                if unexpected_missing:
                    raise RuntimeError(
                        f"Unexpected missing keys in fine-tune checkpoint load: {unexpected_missing}"
                    )
            else:
                log.info(f"  Missing (randomly initialized): {missing}")
                log.info(f"  Unexpected (ignored): {unexpected}")
 
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

            # Model checkpoints — override dirpath so ckpts land in the timestamped run dir
            ckpt_kwargs = OmegaConf.to_container(self._exp_cfg.checkpointer, resolve=True)
            ckpt_kwargs["dirpath"] = ckpt_dir
            callbacks.append(ModelCheckpoint(**ckpt_kwargs))

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

        if self._is_finetune:
            log.info("Running test on best checkpoint")
            trainer.test(
                model=self._model,
                datamodule=self._datamodule,
                ckpt_path="best",
            )

@hydra.main(version_base=None, config_path="./configs", config_name="train_afdb")
def main(cfg: DictConfig):

    exp = Experiment(cfg=cfg)
    exp.train()

if __name__ == "__main__":
    main()