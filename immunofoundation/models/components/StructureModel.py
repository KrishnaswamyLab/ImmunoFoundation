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

    def forward(self, adj, coords, **kwargs):
        coords = self.projection(coords)
        if adj.dim() == 3:
            batch_size, seq_len, _ = adj.shape
            eye = torch.eye(seq_len, device=adj.device, dtype=adj.dtype).unsqueeze(0)
            adj = torch.clamp(adj + eye, max=1)
            attn_mask = torch.zeros_like(adj, dtype=coords.dtype)
            attn_mask = attn_mask.masked_fill(adj == 0, float('-inf'))
            attn_mask = attn_mask.repeat_interleave(self.cfg.n_heads, dim=0)
            return self.encoder(coords, mask=attn_mask)
        return self.encoder(coords)