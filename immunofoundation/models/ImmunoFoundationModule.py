import torch
import time
import os
import random
import wandb
import numpy as np
import pandas as pd
import logging
from pytorch_lightning import LightningModule

from immunofoundation.models.components.ESMSequenceModel import ESMSequenceModel
from immunofoundation.models.components.SequenceModel import SequenceModel
from immunofoundation.models.components.StructureModel import StructureModel
from immunofoundation.models.components.BiochemicalModel import BiochemicalModel

class ImmunoFoundationModule(LightningModule):

    def __init__(self,model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.sequence_model = ESMSequenceModel(model_cfg.sequence)
        self.structure_model = StructureModel(model_cfg.structure)
        self.bio_model = BiochemicalModel(model_cfg.bio_chem)
    
    def training_step(self,batch,stage):

        peptide_seq_embeddings, mhc_seq_embeddings = self.sequence_model(batch['peptide_sequence'],batch['mhc_sequence'])
        peptide_struct_embeddings, mhc_struct_embeddings = self.structure_model(batch['peptide_coords'], batch['mhc_coords'])
        bio_chem_embeddings = self.bio_model(batch['biochemical_properties'])
        return peptide_seq_embeddings, mhc_seq_embeddings, peptide_struct_embeddings, mhc_struct_embeddings, bio_chem_embeddings
    
    def configure_optimizers(self):
        # Use optimizer config if provided in model_cfg, otherwise default
        opt_cfg = getattr(self.model_cfg, 'optimizer', None)
        if opt_cfg is None:
            return torch.optim.AdamW(self.parameters(), lr=1e-4)
        # expect opt_cfg to be a namespace/dict compatible with torch.optim.AdamW args
        kwargs = {}
        if hasattr(opt_cfg, 'lr'):
            kwargs['lr'] = opt_cfg.lr
        if hasattr(opt_cfg, 'weight_decay'):
            kwargs['weight_decay'] = opt_cfg.weight_decay
        return torch.optim.AdamW(self.parameters(), **kwargs)