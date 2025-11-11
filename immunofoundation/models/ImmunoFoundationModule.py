import torch
import time
import os
import random
import wandb
import numpy as np
import pandas as pd
import logging
from pytorch_lightning import LightningModule

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.mlm_decoder = nn.Linear(model_cfg.sequence.out_dim, 20)
        self.sequence_decoder = self.build_decoder(model_cfg.sequence.out_dim, model_cfg.sequence.out_dim*2, model_cfg.sequence.esm_dim)
        self.structure_decoder = self.build_decoder(model_cfg.structure.out_dim, int(model_cfg.structure.out_dim//2), 3)
        self.biochem_decoder = self.build_decoder(model_cfg.bio_chem.out_dim, model_cfg.bio_chem.out_dim*2, model_cfg.bio_chem.n_bio_prop)
        self.mlm_criterion = nn.CrossEntropyLoss(reduction='none')

    def training_step(self,batch,stage):
        per_sample_losses = self.model_step(batch)
        total_losses = {k: v.mean() for k,v in per_sample_losses.items()}
        return sum(total_losses.values())
        
    def model_step(self, batch):
        with torch.no_grad():
            peptide_seq_embeddings, mhc_seq_embeddings, peptide_tokens, mhc_tokens = self.aa_embedding_model(batch['peptide_sequence'],batch['mhc_sequence'], True)
        masked_peptide_seq_embeddings, masked_mhc_seq_embeddings, \
            masked_peptide_coords, masked_mhc_coords = self.mask_residues(peptide_seq_embeddings, mhc_seq_embeddings, batch)
        peptide_seq_embeddings_with_mask, mhc_seq_embeddings_with_mask = self.sequence_model(masked_peptide_seq_embeddings, masked_mhc_seq_embeddings)
        peptide_struct_embeddings, mhc_struct_embeddings = self.structure_model(batch['peptide_adjs'], batch['mhc_adjs'], masked_peptide_coords, masked_mhc_coords)
        bio_chem_embeddings = self.bio_model(batch['biochemical_properties'])

        recon_peptide_seq_embeddings = self.sequence_decoder(peptide_seq_embeddings_with_mask)
        recon_mhc_seq_embeddings = self.sequence_decoder(mhc_seq_embeddings_with_mask)
        recon_peptide_sturct_embeddings = self.structure_decoder(peptide_struct_embeddings)
        recon_mhc_sturct_embeddings = self.structure_decoder(mhc_struct_embeddings)
        recon_bio_chem = self.biochem_decoder(bio_chem_embeddings)

        predicted_peptides_residues = self.mlm_decoder(peptide_seq_embeddings_with_mask)
        predicted_mhc_residues = self.mlm_decoder(mhc_seq_embeddings_with_mask)
        per_sample_losses = {}

        per_sample_losses['recon_loss_peptide_seq'] = self.mae_loss(peptide_seq_embeddings, recon_peptide_seq_embeddings, batch['peptide_masks'])
        per_sample_losses['recon_loss_mhc_seq'] = self.mae_loss(mhc_seq_embeddings, recon_mhc_seq_embeddings, batch['mhc_masks'])
        per_sample_losses['recon_loss_peptide_struct'] = self.mae_loss(batch['peptide_coords'], recon_peptide_sturct_embeddings, batch['peptide_masks'])
        per_sample_losses['recon_loss_mhc_struct'] = self.mae_loss(batch['mhc_coords'], recon_mhc_sturct_embeddings, batch['mhc_masks'])
        per_sample_losses['recon_loss_bio_chem'] = (recon_bio_chem - batch['biochemical_properties']).pow(2).mean(1)
        per_sample_losses['mlm_loss_peptides'] = self.mlm_loss(predicted_peptides_residues, peptide_tokens)
        per_sample_losses['mlm_loss_mhc'] = self.mlm_loss(predicted_mhc_residues, mhc_tokens)
        return per_sample_losses

    def encode(self, batch):
        peptide_seq_embeddings, mhc_seq_embeddings = self.aa_embedding_model(batch['peptide_sequence'],batch['mhc_sequence'])
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
    
    def mae_loss(self, x, x_hat, mask):
        masked = mask.bool()
        per_token_mse = (x_hat - x).pow(2).mean(dim=-1) * masked / masked.sum().clamp_min(1)
        return per_token_mse
    
    def mlm_loss(self, logits, true):
        loss_per_token = self.mlm_criterion(logits.view(-1, logits.size(-1)), true.view(-1))
        return loss_per_token.view(logits.size(0), logits.size(1)).sum(1)

    def build_decoder(self, input_dim, hidden_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  
            nn.ReLU(),           
            nn.Linear(hidden_dim, hidden_dim),  
            nn.ReLU(),           
            nn.Linear(hidden_dim, output_dim),   
        )
