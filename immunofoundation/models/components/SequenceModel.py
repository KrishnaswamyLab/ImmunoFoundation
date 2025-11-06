import torch
import torch.nn as nn
import torch.nn.functional as F

class StructureModel(nn.Module):
    def __init__(self, struc_cfg):
        super().__init__()
        self.cfg = struc_cfg
        self.projection = nn.Linear(3, self.cfg.out_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.cfg.out_dim, nhead=self.cfg.n_heads, dim_feedforward=self.cfg.dim_ffn, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.cfg.n_layers)

    def forward(self, peptide_adj, mhc_adj, peptide_coords, mhc_coords):
        projected_peptide_coords = self.projection(peptide_coords)
        projected_mhc_coords = self.projection(mhc_coords)
        return self.encoder(projected_peptide_coords), self.encoder(projected_mhc_coords)

class SequenceModel(nn.Module):
    """Simple wrapper around Facebook ESM models.

    Expects token tensors (either integer token ids or one-hot) for peptides and MHC.
    The model will convert tokens to strings using `vocab` from config and pass
    them to a pretrained ESM model. The resulting sequence representation is
    projected to `out_dim`.
    """

    def __init__(self, seq_cfg):
        super().__init__()
        self.cfg = seq_cfg
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.cfg.esm_dim, nhead=self.cfg.n_heads, dim_feedforward=self.cfg.dim_ffn, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.cfg.n_layers)
        self.projection = nn.Linear(self.cfg.esm_dim, self.cfg.out_dim)

    def forward(self, peptide_embeddings, mhc_embeddings):
        peptide_embeddings = self.encoder(peptide_embeddings)
        mhc_embeddings = self.encoder(mhc_embeddings)

        return self.projection(peptide_embeddings), self.projection(mhc_embeddings)