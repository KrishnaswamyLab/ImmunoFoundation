import torch
import time
import os
import random
import wandb
import numpy as np
import pandas as pd
import logging
from pytorch_lightning import LightningModule

from immunofoundation.models.components.ESM import ESM
from immunofoundation.models.components.SequenceModel import SequenceModel
from immunofoundation.models.components.StructureModel import StructureModel
from immunofoundation.models.components.BiochemicalModel import BiochemicalModel

SEQUENCE_MODELS = {
    "esm": ESM,
}
STRUCTURE_MODELS = {
    "transformer": StructureModel,
}
BIOCHEM_MODELS = {
    "mlp": BiochemicalModel,
}

class ImmunoFoundationModule(LightningModule):

    def __init__(self,model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.aa_embedding_model = SEQUENCE_MODELS.get(model_cfg.sequence.model_type, 'esm')(model_cfg.sequence)
        self.sequence_model = SequenceModel(model_cfg.sequence)
        self.structure_model = STRUCTURE_MODELS.get(model_cfg.structure.model_type, 'transformer')(model_cfg.structure)
        self.bio_model = BIOCHEM_MODELS.get(model_cfg.bio_chem.model_type, 'mlp')(model_cfg.bio_chem)
    
    def training_step(self,batch,stage):
        peptide_seq_embeddings, mhc_seq_embeddings, peptide_struct_embeddings, mhc_struct_embeddings, \
            bio_chem_embeddings, peptide_seq_embeddings_with_mask, mhc_seq_embeddings_with_mask = self.forward(batch, True)

    def forward(self, batch, mask = False):
        peptide_seq_embeddings, mhc_seq_embeddings = self.aa_embedding_model(batch['peptide_sequence'],batch['mhc_sequence'])
        if(mask):
            masked_peptide_seq_embeddings, masked_mhc_seq_embeddings, \
                masked_peptide_coords, masked_mhc_coords = self.mask_residues(peptide_seq_embeddings, mhc_seq_embeddings, batch)
            peptide_seq_embeddings_with_mask, mhc_seq_embeddings_with_mask = self.sequence_model(masked_peptide_seq_embeddings, masked_mhc_seq_embeddings)
            peptide_struct_embeddings, mhc_struct_embeddings = self.structure_model(batch['peptide_adjs'], batch['mhc_adjs'], masked_peptide_coords, masked_mhc_coords)
            bio_chem_embeddings = self.bio_model(batch['biochemical_properties'])
            return peptide_seq_embeddings, mhc_seq_embeddings, peptide_struct_embeddings, mhc_struct_embeddings, bio_chem_embeddings, peptide_seq_embeddings_with_mask, mhc_seq_embeddings_with_mask
        else:
            peptide_seq_embeddings, mhc_seq_embeddings = self.sequence_model(peptide_seq_embeddings, mhc_seq_embeddings)
            peptide_struct_embeddings, mhc_struct_embeddings = self.structure_model(batch['peptide_adjs'], batch['mhc_adjs'], batch['peptide_coords'], batch['mhc_coords'])
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

    def mask_residues(self, peptide_seq_embeddings, mhc_seq_embeddings, batch):
        masked_peptide_seq_embeddings = peptide_seq_embeddings*(1-batch['peptide_masks']).unsqueeze(-1)
        masked_mhc_seq_embeddings = mhc_seq_embeddings*(1-batch['mhc_masks']).unsqueeze(-1)
        masked_peptide_coords = batch['peptide_coords']*(1-batch['peptide_masks']).unsqueeze(-1)
        masked_mhc_coords = batch['mhc_coords']*(1-batch['mhc_masks']).unsqueeze(-1)
        return masked_peptide_seq_embeddings, masked_mhc_seq_embeddings, masked_peptide_coords, masked_mhc_coords
    