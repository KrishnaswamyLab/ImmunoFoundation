import torch
import torch.nn as nn



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