import torch
import time
import os
import random
import wandb
import numpy as np
import pandas as pd
import logging
from pytorch_lightning import LightningModule

from immunofoundation.models.components.SequenceModel import SequenceModel
from immunofoundation.models.components.StructureModel import StructureModel
from immunofoundation.models.components.BiochemicalModel import BiochemicalModel


class ImmunoFoundationModule(LightningModule):

    def __init__(self,model_cfg):
        super().__init__()
        self.sequence_model = SequenceModel(model_cfg.sequence)

        print(model_cfg.sequence)
        print('ImmunoModel')
    
    def loss_sequence_model(self,embeddings):

        mse_loss = [10.0] #placeholder

        return mse_loss
    
    def training_step(self,batch,stage):

        seq_embeddings = self.sequence_model(batch['peptide_sequence'],batch['mhc_sequence'])

        loss = self.loss_sequence_model(seq_embeddings)

        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(
            params=self.model.parameters(),
            **self._exp_cfg.optimizer
        )